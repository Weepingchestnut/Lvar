import argparse
import copy

import cv2
import numpy as np
import pandas as pd
import torch.distributed as tdist
from lightning_fabric import seed_everything
from PIL import Image
from tqdm import trange
from transformers import AutoModel

from models.hart.hart_transformer_t2i import HARTForT2I
from tools.run_hart import gen_one_img_hart
from tools.run_infinity import *


def create_image_grid(images):
    """
    Stitch 4 images together into a 2x2 grid image.
    Supports image input in two formats: PIL.Image and Numpy Array.
    """
    if not images or len(images) != 4:
        raise ValueError("4 images are required to create a 2x2 grid.")

    # Check the image format and convert them uniformly to PIL.Image.
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            # if BGR (cv2.imwrite default format)
            pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        elif isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            raise TypeError(f"Unsupported image formats: {type(img)}")

    width, height = pil_images[0].size      # assume that all images are the same size
    
    # Create a new 2x2 canvas
    grid_image = Image.new('RGB', (width * 2, height * 2))
    
    # Paste the image into the grid.
    grid_image.paste(pil_images[0], (0, 0))
    grid_image.paste(pil_images[1], (width, 0))
    grid_image.paste(pil_images[2], (0, height))
    grid_image.paste(pil_images[3], (width, height))
    
    return grid_image


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--test_speed', type=bool, default=True, help="Enable latency measurement")
    args = parser.parse_args()

    # ensure n_samples = 4
    if args.n_samples != 4:
        print(f"Warning: DPG-Bench evaluation requires 4 images per prompt, \
              but n_samples is set to {args.n_samples}. Forcing n_samples to 4.")
        args.n_samples = 4

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
    
    # load DPG-Bench prompts
    # dpg_bench_file = 'evaluation/dpg_bench/dpg_bench.csv'
    # try:
    #     df = pd.read_csv(dpg_bench_file)
    #     prompts_df = df[['item_id', 'text']].drop_duplicates().reset_index(drop=True)
    # except FileNotFoundError:
    #     print(f"Error: Cannot find the DPG-Bench CSV file at '{args.dpg_bench_csv}'")
    #     exit()
    
    # *Load DPG-Bench prompts DataFrame on all processes
    prompts_df = None
    if rank == 0: # Let rank 0 check for the file first to provide a clear error message
        dpg_bench_file = 'evaluation/dpg_bench/dpg_bench.csv'
        try:
            prompts_df = pd.read_csv(dpg_bench_file)
            prompts_df = prompts_df[['item_id', 'text']].drop_duplicates().reset_index(drop=True)
        except FileNotFoundError:
            print(f"Error: Cannot find the DPG-Bench CSV file at '{dpg_bench_file}'")
            # Signal other processes to exit gracefully if file is not found
            prompts_df = "FILE_NOT_FOUND"

    # Broadcast the DataFrame (or error signal) from rank 0
    object_list_to_broadcast = [prompts_df]
    tdist.broadcast_object_list(object_list_to_broadcast, src=0)
    prompts_df = object_list_to_broadcast[0]

    if isinstance(prompts_df, str) and prompts_df == "FILE_NOT_FOUND":
        # If rank 0 failed to find the file, all processes exit.
        if rank != 0: print(f"Rank {rank} exiting because DPG-Bench file was not found by rank 0.")
        return

    # *Distribute the DataFrame indices across all GPUs
    total_samples = len(prompts_df)
    per_gpu = (total_samples + world_size - 1) // world_size
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total_samples)
    
    if args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
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

    # Create main output directory and synchronize
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Output directory: {args.outdir}")
    tdist.barrier(device_ids=[local_rank])

    # hyperparameter setting
    tau = args.tau; cfg = args.cfg
    h_div_w_template = 1.000
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    # tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']     # no use
    
    local_total_latency = 0.0
    local_infinity_latency = 0.0
    local_num_images = 0
    warmup_steps = 2
    local_warmup_images = warmup_steps * args.n_samples
    
    # for index, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
    for index in trange(start_idx, end_idx, disable=rank != 0, desc=f"Rank {rank}"):
        seed_everything(args.seed)
        row = prompts_df.iloc[index]
        prompt = row['text']
        item_id = row['item_id']
        
        # Check whether it has been generated, and skip if it exists
        save_file = os.path.join(args.outdir, f"{item_id}.png")
        if os.path.exists(save_file):
            print(f"[Rank {rank}] Skipping [{item_id}], file already exists.")
            continue

        print(f"\n[Rank {rank}] Processing ({index+1: >4}/{len(prompts_df)}): item_id='{item_id}'")
        print(f"[Rank {rank}] Prompt: '{prompt}'")

        images = []
        for sample_j in range(args.n_samples):
            print(f"[Rank {rank}] Generating sample {sample_j+1}/{args.n_samples}...")
            seed = args.seed + (index * args.n_samples) + sample_j
            
            torch.cuda.reset_peak_memory_stats(device=device)
            alloc_before_gen = torch.cuda.memory_allocated(device=device) / (1024**2)
            t1 = time.time()

            # ------------ Infinity ------------
            if 'infinity' in args.model_type:
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
            print(f"\n====== [Rank {rank}] Generation Time / Memory Usage of Original Model ======")
            print(f'{args.model_type} infer one image takes {t2-t1:.2f}s (w/ decoder)')
            print(f"GPU allocated before/after: {alloc_before_gen:.1f} MB -> {alloc_after_gen:.1f} MB (delta {alloc_after_gen - alloc_before_gen:+.1f} MB)")
            print(f"GPU peak allocated during gen: {peak_alloc_gen:.1f} MB (delta {peak_alloc_gen - alloc_before_gen:+.1f} MB)")
            print("=======================================================================\n")
            images.append(image.cpu().numpy())

        # Image gridding and saving
        if len(images) == args.n_samples:
            print(f"[Rank {rank}]  Gridding {len(images)} images into a 2x2 format...")
            grid_image = create_image_grid(images)
            
            # DPG-Bench requires that the file name be the same as the file name in the prompt (i.e., item_id).
            grid_image.save(save_file)
            print(f"[Rank {rank}]  Saved grid image to: {save_file}")
        else:
            print(f"[Rank {rank}] Warning: Expected {args.n_samples} images but got {len(images)}. \
                  Skipping grid creation for item_id '{item_id}'.")

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
    global_warmup_images = local_warmup_images * world_size
    global_num_images = int(latency_tensor[2].item()) - global_warmup_images

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
        
        print("\nDPG-Bench inference finished.")


if __name__ == '__main__':
    main()
