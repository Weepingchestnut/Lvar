import argparse
import copy
import os
import os.path as osp
import re
import time

import cv2
import hpsv2
import torch
import torch.distributed as tdist
from pytorch_lightning import seed_everything
# from diffusers import FluxPipeline
from tqdm import trange
from transformers import AutoModel

from models.hart.hart_transformer_t2i import HARTForT2I
from tools.conf import HF_HOME, HF_TOKEN
from tools.run_hart import gen_one_img_hart
from tools.run_infinity import *

# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--test_speed', type=bool, default=True, help="Enable latency measurement")
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # *Initialize distributed process group
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    tdist.init_process_group(backend='nccl')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()

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
    # elif args.model_type == 'flux_1_dev':
    #     pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    # elif args.model_type == 'flux_1_dev_schnell':
    #     pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        print(f"[Rank {rank}] Loading Infinity model...")
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)
    elif 'hart' in args.model_type:
        print(f"[Rank {rank}] Loading HART model...")
        # load text encoder
        text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_ckpt)
        text_encoder = AutoModel.from_pretrained(args.text_encoder_ckpt).to(device)
        text_encoder.eval()
        # load hart
        hart = AutoModel.from_pretrained(args.model_path)
        hart = hart.to(device)
        hart.eval()
        if args.use_ema:
            ema_hart = copy.deepcopy(hart)
            ema_hart.load_state_dict(torch.load(os.path.join(args.model_path, "ema_model.bin")))

    # Create main output directory and synchronize
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Output directory: {args.outdir}")
    tdist.barrier(device_ids=[local_rank])

    # Infinity hyperparameter setting
    tau = args.tau; cfg = args.cfg
    h_div_w_template = 1.000
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    # tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']
    
    local_total_latency = 0.0
    local_infinity_latency = 0.0
    local_num_images = 0
    warmup_steps = 5

    for index in trange(start_idx, end_idx, disable=rank != 0, desc=f"Rank {rank}"):
        seed_everything(args.seed)
        
        task_info = flat_prompts_list[index]
        style = task_info["style"]
        original_idx = task_info["original_idx"]
        prompt = task_info["prompt"]

        if rank == 0 and index % 10 == 0:
            print(f'Generate {index}/{total_samples} images...')
        
        image_save_file_path = os.path.join(args.outdir, style, f"{original_idx:05d}.jpg")
        os.makedirs(osp.dirname(image_save_file_path), exist_ok=True)
        
        images = []
        for sample_j in range(args.n_samples):
            print(f"[Rank {rank}] Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            # Important! for reproducibility, e.g. 4 samples of same prompt will same
            seed = args.seed + (index * args.n_samples) + sample_j

            torch.cuda.reset_peak_memory_stats(device=device)
            alloc_before_gen = torch.cuda.memory_allocated(device=device) / (1024**2)
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
            # ------------ Infinity ------------
            elif 'infinity' in args.model_type:
                if args.test_speed and index > (warmup_steps-1):
                    image, total_cost, infinity_cost = gen_one_img(
                        infinity, vae, text_tokenizer, text_encoder,
                        prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg,
                        scale_schedule=scale_schedule,
                        cfg_insertion_layer=[args.cfg_insertion_layer],
                        vae_type=args.vae_type,
                        g_seed=seed,
                        test_speed=args.test_speed
                    )
                    local_total_latency += total_cost
                    local_infinity_latency += infinity_cost
                    local_num_images += 1
                else:
                    image = gen_one_img(
                        infinity, vae, text_tokenizer, text_encoder,
                        prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg,
                        scale_schedule=scale_schedule,
                        cfg_insertion_layer=[args.cfg_insertion_layer],
                        vae_type=args.vae_type,
                        g_seed=seed,
                    )
            # ------------ HART ------------
            elif 'hart' in args.model_type:
                image = gen_one_img_hart(
                    hart, args.use_ema, ema_hart, text_tokenizer, text_encoder,
                    prompt, cfg, args.max_token_length, args.use_llm_system_prompt,
                    args.more_smooth,
                )
            else:
                raise ValueError
            t2 = time.time()

            alloc_after_gen = torch.cuda.memory_allocated(device=device) / (1024**2)
            peak_alloc_gen = torch.cuda.max_memory_allocated(device=device) / (1024**2)
            print(f"\n===== [Rank {rank}] Generation Time / Memory Usage of Original Model =====")
            print(f'{args.model_type} infer one image takes {t2-t1:.2f}s (w/ decoder)')
            print(f"GPU allocated before/after: {alloc_before_gen:.1f} MB -> {alloc_after_gen:.1f} MB (delta {alloc_after_gen - alloc_before_gen:+.1f} MB)")
            print(f"GPU peak allocated during gen: {peak_alloc_gen:.1f} MB (delta {peak_alloc_gen - alloc_before_gen:+.1f} MB)")
            print("=======================================================================\n")
            
            images.append(image)
        
        assert len(images) == 1, f"HPSv2.1 typically samples only one image per prompt. Please set n_samples to 1."
        for index, image in enumerate(images):
            if 'infinity' in args.model_type:
                cv2.imwrite(image_save_file_path, image.cpu().numpy())
            else:
                image.save(image_save_file_path)
    tdist.barrier(device_ids=[local_rank])
    
    # ---- distributed latency reduction ----
    latency_tensor = torch.tensor(
        [local_total_latency, local_infinity_latency, local_num_images],
        device=device,
        dtype=torch.float64
    )
    
    tdist.all_reduce(latency_tensor, op=tdist.ReduceOp.SUM)
    global_total_latency = latency_tensor[0].item()
    global_infinity_latency = latency_tensor[1].item()
    global_num_images = int(latency_tensor[2].item())

    if rank == 0:
        avg_total_latency = global_total_latency / max(global_num_images, 1)
        avg_infinity_latency = global_infinity_latency / max(global_num_images, 1)
        throughput_total = global_num_images / global_total_latency
        throughput_infinity = global_num_images / global_infinity_latency
        warmup_images = warmup_steps * args.n_samples
        print("\n======== Benchmark Latency ========")
        print(f"Total images: {global_num_images} + {warmup_images} = {global_num_images+warmup_images}")
        print("------ w/ decoder ------")
        print(f"Total inference time: {global_total_latency:.2f} s")
        print(f"Average latency per image: {avg_total_latency:.4f} s")
        print(f"Throughput: {throughput_total:.4f} images/sec")
        print("------ w/o decoder ------")
        print(f"Total inference time: {global_infinity_latency:.2f} s")
        print(f"Average latency per image: {avg_infinity_latency:.4f} s")
        print(f"Throughput: {throughput_infinity:.4f} images/sec")
        print("===================================\n")
        
        print("All ranks finished generation. Starting evaluation...")
        hpsv2.evaluate(args.outdir, hps_version="v2.1")
        print("Evaluation finished.")


if __name__ == '__main__':
    main()
