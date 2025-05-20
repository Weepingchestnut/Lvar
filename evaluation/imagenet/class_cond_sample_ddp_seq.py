import argparse
import math
import os
import random
import time
import numpy as np
import torch_fidelity
import torch
import torch.nn as nn
import torch.distributed as tdist
from tqdm import tqdm
from PIL import Image

from models.vae.vqvae import VQVAE
from models.var.var_model import VAR


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """

    # run faster
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)  # True: fast but may lead to some small numerical differences
    torch.set_float32_matmul_precision('high' if args.tf32 else 'highest')

    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    # torch.set_grad_enabled(False)

    # ------ Setup DDP: ------
    if 'LOCAL_RANK' not in os.environ: # Manual DDP launch without torchrun/slurm
        # For manual launch, ensure WORLD_SIZE and RANK are set per process
        # This part might need adjustment based on how you launch multiple processes
        if 'WORLD_SIZE' not in os.environ: os.environ['WORLD_SIZE'] = str(torch.cuda.device_count()) # Guess
        if 'RANK' not in os.environ: os.environ['RANK'] = '0' # Master address needed for others
        if 'MASTER_ADDR' not in os.environ: os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ: os.environ['MASTER_PORT'] = '29500' # Example port
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # NCCL backend is recommended for multi-GPU training on NVIDIA GPUs.
    tdist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=global_rank)

    # Set device AFTER init_process_group if using local_rank directly for device mapping
    # In DDP, each process typically manages one GPU.
    device = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    # seed
    base_seed = args.global_seed
    seed = base_seed + global_rank # Simple way to differentiate seeds
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)      # DiT code not need
    random.seed(seed)
    np.random.seed(seed)
    # make sure cudnn determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if global_rank == 0:
        print(f"World size: {world_size}. \
              Process {global_rank} (local {local_rank}) on device cuda:{device}. \
                Global seed: {args.global_seed}")
    
    # ------ TARGETS FOR GENERATION ------
    target_images_per_class = 50
    # Update num_fid_samples based on the new requirement
    args.num_fid_samples = args.num_classes * target_images_per_class
    if global_rank == 0:
        print(f"Generating {target_images_per_class} images for each of {args.num_classes} classes.")
        print(f"Total images to generate: {args.num_fid_samples}")

    # ------ Load model ------
    assert args.model_depth in {16, 20, 24, 30}
    vae_ckpt, var_ckpt = 'pretrained_models/var/vae_ch160v4096z32.pth', f'pretrained_models/var/var_d{args.model_depth}.pth'

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    heads = args.model_depth
    width = args.model_depth * 64
    dpr = 0.1 * args.model_depth/24

    # disable built-in initialization for speed
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)

    # build models
    vae = VQVAE(
        vocab_size=4096, 
        z_channels=32, 
        ch=160, 
        test_mode=True, 
        share_quant_resi=4, 
        v_patch_nums=patch_nums).to(device)
    
    var = VAR(
        vae_local=vae,
        num_classes=1000, depth=args.model_depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=True,
        patch_nums=patch_nums,
        flash_if_available=True, fused_if_available=True,
    ).to(device)
    var.init_weights(init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1)    # init_std < 0: automated

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu', weights_only=True), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu', weights_only=True), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    
    if global_rank == 0:
        print(f'Model prepare finished.')

    # Create folder to save samples:
    model_string_name = "VAR-d{}".format(args.model_depth)
    folder_name = "{}-size{}_cfg-{}_seed-{}".format(
        model_string_name, args.image_size, args.cfg_scale, args.global_seed)
    sample_folder_dir = os.path.join(args.work_dir, folder_name)
    if global_rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    tdist.barrier(device_ids=[device] if torch.cuda.is_available() else None)

    # ------ Determine samples per GPU for each class ------
    # Each GPU will generate a portion of the 50 images for each class.
    # This ensures work is distributed.
    samples_per_gpu_for_a_class = [0] * world_size
    base_count = target_images_per_class // world_size
    remainder = target_images_per_class % world_size
    for i in range(world_size):
        samples_per_gpu_for_a_class[i] = base_count + (1 if i < remainder else 0)
    
    my_samples_for_any_given_class = samples_per_gpu_for_a_class[global_rank]

    # Calculate the starting global index for samples generated by this rank for any class
    # This is to ensure that file names are unique and sequential across all classes and ranks
    my_sample_offset_within_a_class = sum(samples_per_gpu_for_a_class[i] for i in range(global_rank))

    if global_rank == 0:
        print(f"Distribution of samples per class per GPU: {samples_per_gpu_for_a_class}")
    print(f"Rank {global_rank} will generate {my_samples_for_any_given_class} samples for each class.")

    # ------ Main Generation Loop ------
    # tqdm setup for rank 0 to show total images this rank will generate
    total_images_this_gpu_will_generate = args.num_classes * my_samples_for_any_given_class
    pbar = None
    if global_rank == 0:
        pbar = tqdm(total=total_images_this_gpu_will_generate, desc=f"Rank 0 Generating {total_images_this_gpu_will_generate} samples")

    generation_start_time = time.time()

    for class_idx in range(args.num_classes):
        num_generated_by_me_for_this_class = 0
        
        if global_rank == 0 and pbar is not None:
             pbar.set_description(f"Rank 0 Gen Class {class_idx+1}/{args.num_classes}")

        while num_generated_by_me_for_this_class < my_samples_for_any_given_class:
            # Determine batch size for current step for this GPU
            # args.per_proc_batch_size is 'n' in your original code
            current_gpu_batch_size = min(
                args.per_proc_batch_size,
                my_samples_for_any_given_class - num_generated_by_me_for_this_class
            )
            if current_gpu_batch_size == 0: # Should not happen if logic is correct
                break

            # Prepare class labels: all are the current class_idx for this batch
            label_B = torch.full((current_gpu_batch_size,), class_idx, device=device, dtype=torch.long)

            current_g_seed = seed + class_idx # Simple variation for generation

            with torch.inference_mode():
                with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):     # using bfloat16 can be faster
                    recon_B3HW = var.autoregressive_infer_cfg(
                        B=current_gpu_batch_size,
                        label_B=label_B,
                        cfg=args.cfg_scale,
                        top_k=900,
                        top_p=0.96,
                        g_seed=current_g_seed,
                        more_smooth=args.more_smooth)
            
            # Process and save samples
            recon_BHW3 = torch.clamp(recon_B3HW.mul(255), 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i in range(current_gpu_batch_size):
                # Calculate the global sequential index for the image file
                # This is the index of the sample within the current class, across all GPUs
                instance_idx_within_class = my_sample_offset_within_a_class + num_generated_by_me_for_this_class + i
                
                # This is the overall global image index across all classes
                global_image_idx = (class_idx * target_images_per_class) + instance_idx_within_class
                
                Image.fromarray(recon_BHW3[i]).save(f"{sample_folder_dir}/{global_image_idx:06d}.png")

            num_generated_by_me_for_this_class += current_gpu_batch_size
            if global_rank == 0 and pbar is not None:
                pbar.update(current_gpu_batch_size)

    if global_rank == 0 and pbar is not None:
        pbar.close()
    
    total_generation_time = time.time() - generation_start_time
    if global_rank == 0:
        print(f"Finished generating all samples. Total time: {total_generation_time:.2f} seconds.")
        print(f"Average time per image (global): {total_generation_time / args.num_fid_samples:.4f} seconds.")

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    tdist.barrier(device_ids=[device] if torch.cuda.is_available() else None)
    if global_rank == 0:
        print(f"Attempting to create .npz file from {args.num_fid_samples} samples...")
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("create npz Done.")

        # ------ torch compute FID, IS ------
        # get current .py path
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        if args.image_size == 256:
            input2 = None
            fid_statistics_file = os.path.join(script_dir, 'adm_in256_stats.npz')
        else:
            raise NotImplementedError
        
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=sample_folder_dir,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        print("\nFID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        # -----------------------------------
    
    tdist.barrier(device_ids=[device] if torch.cuda.is_available() else None)
    tdist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-depth", type=int, choices=[16, 20, 24, 30], default=16)
    # parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    # parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--work-dir", type=str, default="work_dir/seq_gen_ddp")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--cfg-scale",  type=float, default=1.5,
                        help="Note a relatively small cfg=1.5 is used for trade-off between image quality and diversity. \
                            You can adjust it to cfg=5.0 for better visual quality.")

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--more-smooth", action=argparse.BooleanOptionalAction, default=False,
                        help="For better visual quality")
    # parser.add_argument("--ckpt", type=str, default=None,
    #                     help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()

    main(args)
