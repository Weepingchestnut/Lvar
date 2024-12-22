import argparse
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as tdist
from tqdm import tqdm
from PIL import Image

from models.vae.vqvae import VQVAE
from models.var.var import VAR


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
    tdist.init_process_group("nccl")
    rank = tdist.get_rank()
    device = rank % torch.cuda.device_count()

    # seed
    seed = args.global_seed * tdist.get_world_size() + rank
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)      # DiT code not need
    random.seed(seed)
    np.random.seed(seed)
    # make sure cudnn determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={tdist.get_world_size()}.")

    # ------ Load model ------
    assert args.model_depth in {16, 20, 24, 30}
    vae_ckpt, var_ckpt = 'checkpoints/var/vae_ch160v4096z32.pth', f'checkpoints/var/var_d{args.model_depth}.pth'

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
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'Model prepare finished.')

    # Create folder to save samples:
    model_string_name = "VAR-d{}".format(args.model_depth)
    folder_name = "{}-size{}_cfg-{}_seed-{}".format(model_string_name, args.image_size, args.cfg_scale, args.global_seed)
    sample_folder_dir = os.path.join(args.work_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    tdist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    print("n = {}".format(n))
    global_batch_size = n * tdist.get_world_size()
    print("global_batch_size = {}".format(global_batch_size))
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % tdist.get_world_size() == 0, "total_samples must be divisible by world_size"

    samples_needed_this_gpu = int(total_samples // tdist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar

    used_time = 0
    total = 0

    for _ in pbar:
        class_labels = torch.randint(0, args.num_classes, (n,), device=device)
        # label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        label_B: torch.LongTensor = class_labels.long()

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):     # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=n, label_B=label_B, cfg=args.cfg_scale, top_k=900, top_p=0.95, g_seed=seed, more_smooth=args.more_smooth)
            
        # measure speed after the first generation batch
        if total >= global_batch_size:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(total, used_time, used_time / total))

        tdist.barrier()
        recon_BHW3 = torch.clamp(recon_B3HW.mul(255), 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        # Save samples to disk as individual .png files
        for i, sample in enumerate(recon_BHW3):
            index = i * tdist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    tdist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("create npz Done.")
    tdist.barrier()
    tdist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-depth", type=int, choices=[16, 20, 24, 30], default=16)
    # parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    # parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--cfg-scale",  type=float, default=1.5,
                        help="Note a relatively small cfg=1.5 is used for trade-off between image quality and diversity. \
                            You can adjust it to cfg=5.0 for better visual quality.")
    parser.add_argument("--num-sampling-steps", type=int, default=250)

    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--more-smooth", action=argparse.BooleanOptionalAction, default=False,
                        help="For better visual quality")
    # parser.add_argument("--ckpt", type=str, default=None,
    #                     help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()

    main(args)

