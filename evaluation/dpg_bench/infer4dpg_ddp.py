import argparse
import cv2
from lightning_fabric import seed_everything
import numpy as np
import pandas as pd
import torch.distributed as tdist
from PIL import Image
from tqdm import trange

from models.scalekv.scale_kv import enable_scale_kv
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
    tdist.init_process_group(backend='nccl')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()
    torch.cuda.set_device(rank)
    
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

        if 'scalekv' in args.model_type:
            infinity = enable_scale_kv(infinity, window_size=16, max_capacity=650, kernel_size=5, pooling='maxpool')

    # Create main output directory and synchronize
    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Output directory: {args.outdir}")
    tdist.barrier()

    # hyperparameter setting
    tau = args.tau; cfg = args.cfg
    h_div_w_template = 1.000
    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    # tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']     # no use
    
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
            print(f"[Rank {rank}]  Generating sample {sample_j+1}/{args.n_samples}...")
            t1 = time.time()

            image = gen_one_img(
                infinity, vae, text_tokenizer, text_encoder,
                prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type
            )
            
            t2 = time.time()
            print(f'[Rank {rank}] {args.model_type} infer one image takes {t2-t1:.2f}s')
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

    tdist.barrier()
    if rank == 0:
        print("\nDPG-Bench inference finished.")


if __name__ == '__main__':
    main()
