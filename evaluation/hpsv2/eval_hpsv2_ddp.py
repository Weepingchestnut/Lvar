import os
import os.path as osp
import time
import argparse
import re

import cv2
import hpsv2
import torch
import torch.distributed as tdist
from pytorch_lightning import seed_everything
from diffusers import FluxPipeline
from tqdm import trange

from models.scalekv.scale_kv import enable_scale_kv
from tools.run_infinity import *
from tools.conf import HF_TOKEN, HF_HOME

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # *Initialize distributed process group
    tdist.init_process_group(backend='nccl')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()
    torch.cuda.set_device(rank)

    # all_prompts = hpsv2.benchmark_prompts('all')
    # --> Flatten the nested data structure of HPSv2 to facilitate partitioning.
    all_prompts_dict = hpsv2.benchmark_prompts('all')
    flat_prompts_list = []
    for style, prompts in all_prompts_dict.items():
        for idx, prompt in enumerate(prompts):
            flat_prompts_list.append({
                "style": style,
                "original_idx": idx,
                "prompt": prompt
            })
    
    total_samples = len(flat_prompts_list)
    per_gpu = (total_samples + world_size - 1) // world_size
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total_samples)

    seed_everything(args.seed)

    if args.model_type == 'sdxl':
        from diffusers import DiffusionPipeline
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda")

        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
    elif args.model_type == 'sd3':
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    elif args.model_type == 'pixart_sigma':
        from diffusers import PixArtSigmaPipeline
        pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16
        ).to("cuda")
    elif args.model_type == 'flux_1_dev':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

        if 'scalekv' in args.model_type:
            infinity = enable_scale_kv(infinity, window_size=16, max_capacity=650, kernel_size=5, pooling='maxpool')
    
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
    tdist.barrier()

    for i in trange(start_idx, end_idx, disable=rank != 0, desc=f"Rank {rank}"):
        task_info = flat_prompts_list[i]
        style = task_info["style"]
        original_idx = task_info["original_idx"]
        prompt = task_info["prompt"]

        if rank == 0 and i % 10 == 0:
            print(f'Generate {i}/{total_samples} images...')
        
        image_save_file_path = os.path.join(args.outdir, style, f"{original_idx:05d}.jpg")
        os.makedirs(osp.dirname(image_save_file_path), exist_ok=True)

        tau = args.tau
        cfg = args.cfg

        t1 = time.time()
        if args.model_type == 'sdxl':
            image = base(
                prompt=prompt,
                num_inference_steps=40,
                denoising_end=0.8,
                output_type="latent",
            ).images
            image = refiner(
                prompt=prompt,
                num_inference_steps=40,
                denoising_start=0.8,
                image=image,
            ).images[0]
        elif args.model_type == 'sd3':
            image = pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
                num_images_per_prompt=1,
            ).images[0]
        elif args.model_type == 'flux_1_dev':
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                num_images_per_prompt=1,
            ).images[0]
        elif args.model_type == 'flux_1_dev_schnell':
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=0.0,
                num_inference_steps=4,
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
        elif args.model_type == 'pixart_sigma':
            image = pipe(prompt).images[0]
        elif 'infinity' in args.model_type:
            h_div_w_template = 1.000
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
            tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
            image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type
            )
        else:
            raise ValueError
        t2 = time.time()
        print(f'[Rank {rank}] {args.model_type} infer one image takes {t2-t1:.2f}s')
        if 'infinity' in args.model_type:
            cv2.imwrite(image_save_file_path, image.cpu().numpy())
        else:
            image.save(image_save_file_path)
    tdist.barrier()

    if rank == 0:
        print("All ranks finished generation. Starting evaluation...")
        hpsv2.evaluate(args.outdir, hps_version="v2.1")
        print("Evaluation finished.")


if __name__ == '__main__':
    main()
