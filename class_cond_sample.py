import argparse
import os
import random
import numpy as np
import PIL.Image as PImage
import torch
import torch.nn as nn
import torchvision
import tqdm

from models import build_vae_var
from models.vae.vqvae import VQVAE
from models.var.var import VAR
from utils.misc import create_npz_from_sample_folder


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_depth", type=int, default=16)
    parser.add_argument("--cfg", type=int, default=1.5)
    parser.add_argument("--more_smooth", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="work_dir/class_cond_VARd16")
    args = parser.parse_args()


    MODEL_DEPTH = args.model_depth    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}

    # ========== 1. build model ==========
    vae_ckpt, var_ckpt = 'checkpoints/var/vae_ch160v4096z32.pth', f'checkpoints/var/var_d{MODEL_DEPTH}.pth'

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    heads = MODEL_DEPTH
    width = MODEL_DEPTH * 64
    dpr = 0.1 * MODEL_DEPTH/24

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
        num_classes=1000, depth=MODEL_DEPTH, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
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
    print(f'prepare finished.')

    # ========== 2. Sample with classifier-free guidance ==========

    # set args
    seed = 0 #@param {type:"number"}
    torch.manual_seed(seed)
    cfg = args.cfg #@param {type:"slider", min:1, max:10, step:0.1}
    more_smooth = args.more_smooth # True for more smooth output

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # ========== 3. Class condition Sample ==========
    print("#####################sampling begin!#####################")

    for i in tqdm.tqdm(range(1000)):
        # print("class index: {}".format(i))
        class_labels = (i,) * 50
        with torch.inference_mode():
            B = len(class_labels)
            label_B: torch.LongTensor = torch.tensor(class_labels, device=device)

            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):     # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

            chw_list = [recon_B3HW[b].mul_(255).cpu().numpy() for b in range(B)]
            chw_list = [PImage.fromarray(chw.transpose([1, 2, 0]).astype(np.uint8)) for chw in chw_list]
            for b in range(B):
                chw_list[b].save(args.output_path+"/"+str(i)+"_"+str(b)+".PNG")
    
    create_npz_from_sample_folder(args.output_path)
    print("#####################sampling completed!#####################")
