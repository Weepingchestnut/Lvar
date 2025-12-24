import argparse
import copy
import json
import os
import time

import cv2
import torch
import torch.distributed as tdist
from pytorch_lightning import seed_everything
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
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--metadata_file', type=str, default='evaluation/image_reward/benchmark-prompts.json')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--test_speed', type=bool, default=True, help="Enable latency measurement")
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # *Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    tdist.init_process_group(backend='nccl')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()

    # Load metadata only on rank 0 initially to avoid redundant I/O
    if rank == 0:
        with open(args.metadata_file) as fp:
            metadatas = json.load(fp)
        print(f"Loaded {len(metadatas)} prompts from {args.metadata_file}")
    else:
        metadatas = None
    
    # Broadcast the loaded metadata from rank 0 to all other processes
    # This ensures all processes have the same data without reading the file multiple times.
    object_list_to_broadcast = [metadatas]
    tdist.broadcast_object_list(object_list_to_broadcast, src=0)
    metadatas = object_list_to_broadcast[0]

    # *Distribute the data across all GPUs
    total_samples = len(metadatas)
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
    elif args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
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

        if args.rewrite_prompt:
            from tools.prompt_rewriter import PromptRewriter
            prompt_rewriter = PromptRewriter(system='', few_shot_history=[])
    
    # Use barrier to sync before any process starts writing files
    tdist.barrier(device_ids=[local_rank])

    # hyperparameter setting
    tau = args.tau; cfg = args.cfg
    h_div_w_template = 1.000
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    # tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

    # save_metadatas = []
    # -->
    local_save_metadatas = [] # Each process will collect its own results
    
    local_total_latency = 0.0
    local_infinity_latency = 0.0
    local_num_images = 0
    warmup_steps = 2
    local_warmup_images = warmup_steps * args.n_samples
    
    # for index, metadata in enumerate(metadatas):
    for index in trange(start_idx, end_idx, disable=rank != 0, desc=f"Rank {rank}"):
        seed_everything(args.seed)
        metadata = metadatas[index]
        prompt_id = metadata['id']
        prompt = metadata['prompt']

        sample_path = os.path.join(args.outdir, prompt_id)
        os.makedirs(sample_path, exist_ok=True)
        print(f"[Rank {rank}] Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        if args.rewrite_prompt:
            refined_prompt = prompt_rewriter.rewrite(prompt)
            input_key_val = extract_key_val(refined_prompt)
            prompt = input_key_val['prompt']
            print(f'prompt: {prompt}, refined_prompt: {refined_prompt}')
        
        images = []
        # for _ in range(args.n_samples):
        for sample_j in range(args.n_samples):
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
                if args.test_speed:
                    image, total_cost, infinity_cost = gen_one_img(
                        infinity, vae, text_tokenizer, text_encoder,
                        prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg,
                        scale_schedule=scale_schedule,
                        cfg_insertion_layer=[args.cfg_insertion_layer],
                        vae_type=args.vae_type,
                        g_seed=seed,
                        test_speed=args.test_speed
                    )
                    local_num_images += 1
                    if local_num_images > local_warmup_images:
                        local_total_latency += total_cost
                        local_infinity_latency += infinity_cost
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

        # os.makedirs(sample_path, exist_ok=True)
        metadata['gen_image_paths'] = []
        for i, image in enumerate(images):
            save_file_path = os.path.join(sample_path, f"{prompt_id}_{i}.jpg")
            if 'infinity' in args.model_type:
                cv2.imwrite(save_file_path, image.cpu().numpy())
            else:
                image.save(save_file_path)
            metadata['gen_image_paths'].append(save_file_path)
        # print(save_file_path)
        local_save_metadatas.append(metadata)

    # Gather results from all processes to rank 0
    # This is crucial for saving a single, complete metadata file at the end.
    all_processes_metadata = [None] * world_size
    tdist.all_gather_object(all_processes_metadata, local_save_metadatas)
    
    # ---- distributed latency reduction ----
    latency_tensor = torch.tensor(
        [local_total_latency, local_infinity_latency, local_num_images],
        device=device,
        dtype=torch.float64
    )
    
    tdist.all_reduce(latency_tensor, op=tdist.ReduceOp.SUM)
    global_total_latency = latency_tensor[0].item()
    global_infinity_latency = latency_tensor[1].item()
    global_warmup_images = local_warmup_images * world_size
    global_num_images = int(latency_tensor[2].item()) - global_warmup_images

    # Only rank 0 writes the final consolidated metadata file
    if rank == 0:
        avg_total_latency = global_total_latency / max(global_num_images, 1)
        avg_infinity_latency = global_infinity_latency / max(global_num_images, 1)
        throughput_total = global_num_images / global_total_latency
        throughput_infinity = global_num_images / global_infinity_latency
        print("\n========== Benchmark Profile ==========")
        print(f"Total images: {global_num_images} + {global_warmup_images}(warmup) = {global_num_images+global_warmup_images}({total_samples*args.n_samples})")
        print("------ w/ decoder ------")
        print(f"Total inference time: {global_total_latency:.2f} s")
        print(f"Average latency per image: {avg_total_latency:.4f} s")
        print(f"Throughput: {throughput_total:.4f} images/sec")
        print("------ w/o decoder ------")
        print(f"Total inference time: {global_infinity_latency:.2f} s")
        print(f"Average latency per image: {avg_infinity_latency:.4f} s")
        print(f"Throughput: {throughput_infinity:.4f} images/sec")
        print("=======================================\n")
        
        print("All ranks finished generation. Consolidating metadata...")
        # Flatten the list of lists into a single list
        final_save_metadatas = [item for sublist in all_processes_metadata for item in sublist]

        save_metadata_file_path = os.path.join(args.outdir, "metadata.jsonl")
        with open(save_metadata_file_path, "w") as fp:
            json.dump(final_save_metadatas, fp)
            # -->
            # The original saved a list of dicts as a single JSON object. 
            # If it should be one JSON object per line (jsonl), the loop should be:
            # for meta_item in final_save_metadatas:
            #      fp.write(json.dumps(meta_item) + '\n')
        print(f"Consolidated metadata saved to {save_metadata_file_path}")
    tdist.barrier(device_ids=[local_rank])


if __name__ == '__main__':
    main()
