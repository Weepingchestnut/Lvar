import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import torch
import torch.nn as nn
import torchvision
setattr(nn.Linear, "reset_parameters", lambda self: None)
setattr(nn.LayerNorm, "reset_parameters", lambda self: None)
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

from models.vae.vqvae import VQVAE
from models.var.var_model import VAR


if __name__ == '__main__':
    MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
    assert MODEL_DEPTH in {16, 20, 24, 30}

    vae_ckpt, var_ckpt = 'pretrained_models/var/vae_ch160v4096z32.pth', f'pretrained_models/var/var_d{MODEL_DEPTH}.pth'

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if 'vae' not in globals() or 'var' not in globals():
    #     vae, var = build_vae_var(
    #         V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
    #         device=device, patch_nums=patch_nums,
    #         num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    #     )
    
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
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu', weights_only=True), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu', weights_only=True), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    # set args
    seed = 0 #@param {type:"number"}
    num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
    cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}
    class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  #@param {type:"raw"}
    more_smooth = False # True for more smooth output

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

    # sample
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)   # torch.Size([8])

    # ------ torch profile -------------------------
    # with profile(
    #     activities=[
    #         ProfilerActivity.CPU,
    #         ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     on_trace_ready=tensorboard_trace_handler('./logs')
    # ) as prof:
    #     with record_function("model_inference"):
    #         with torch.inference_mode():
    #             with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):     # using bfloat16 can be faster
    #                 recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)
    
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # ----------------------------------------------

    # ------ fvcore Influence ------
    # TODO
    # with torch.inference_mode():
    #     with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):     # using bfloat16 can be faster
    #         print(flop_count_table(FlopCountAnalysis(var.autoregressive_infer_cfg, B), activations=ActivationCountAnalysis(var.autoregressive_infer_cfg, B)))
    # ------------------------------

    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):     # using bfloat16 can be faster
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.96, g_seed=seed, more_smooth=more_smooth)
    
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    chw.save(f"var_d{MODEL_DEPTH}_test.png")





