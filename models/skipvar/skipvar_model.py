import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

import utils.dist as dist
from models.infinity.basic_infinity import (AdaLNBeforeHead, FastRMSNorm,
                                            SelfAttnBlock,
                                            precompute_rope2d_freqs_grid)
from models.infinity.flex_attn import FlexAttn
from models.infinity.infinity_model import (
    TIMM_KEYS, Infinity, MultiInpIdentity, MultipleLayers, SharedAdaLin,
    TextAttentivePool, sample_with_top_k_top_p_also_inplace_modifying_logits_)
from models.skipvar.basic_skipvar import (CrossAttnBlock, flash_attn_func,
                                          flash_fused_op_installed)
from utils.dist import for_visualize
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from models.infinity.fused_op import (fused_ada_layer_norm,
                                          fused_ada_rms_norm)
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


@register_model
def skipvar_infinity_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8, **kwargs):
    return SkipVAR_Infinity(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        block_chunks=block_chunks,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def skipvar_infinity_8b(depth=40, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8, **kwargs):
    return SkipVAR_Infinity(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        block_chunks=block_chunks,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )


class SkipVAR_Infinity(Infinity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.unregistered_blocks = []
        for block_idx in range(self.depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=self.cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=self.shared_aln, norm_layer=self.norm_layer,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, drop=self.drop_rate, drop_path=self.dpr[block_idx], tau=self.tau, cos_attn=self.cos_attn,
                swiglu=self.swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=self.fused_mlp, fused_norm_func=self.fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=self.use_flex_attn, batch_size=self.batch_size, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)

        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))

        import os
        import joblib

        self.decision_models_folder='pretrained_models/skipvar/'+'decision_model_ssim_84/'
        self.skip_model = joblib.load(os.path.join(self.decision_models_folder+"skip_model_Logistic_Regression.pkl"))
        self.skip_scaler = joblib.load(os.path.join(self.decision_models_folder+"skip_model_scaler.pkl"))
        self.uncond_model = joblib.load(os.path.join(self.decision_models_folder+"uncond_model_Logistic_Regression.pkl"))
        self.uncond_scaler = joblib.load(os.path.join(self.decision_models_folder+"uncond_model_scaler.pkl"))

        print(f'\n[SkipVAR], {self.decision_models_folder}')

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        select_model=None
    ):  # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # 如果 forward 中训练时可能随机丢弃条件，这里推理时直接使用条件版（可保留丢弃逻辑也可去掉）
            kv_normal = self.text_norm(kv_compact).contiguous()
            sos_normal = self.text_proj_for_sos((kv_normal, cu_seqlens_k, max_seqlen_k)).float().contiguous()
            kv_normal = self.text_proj_for_ca(kv_normal).contiguous()
            cond_BD_normal = self.shared_ada_lin(sos_normal).contiguous()  # cond_BD 依赖 sos_normal
            kv_step12 = kv_normal.clone()
            ca_kv_step12 = (kv_step12, cu_seqlens_k, max_seqlen_k)
            cond_BD_step12 = cond_BD_normal.clone()

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify,
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT

        if any(np.array(cfg_list) != 1):
            bs = 2 * B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total + le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:] + cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))  # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact)  # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)

        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0:  # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1:  # add cfg on probs
                add_cfg_on_probs = True  # todo in the future, we may want to add cfg on logits and probs
            elif item < 0:  # determine to add cfg at item-th layer's output
                assert leng + item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng + item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')

        num_stages_minus_1 = len(scale_schedule) - 1
        summed_codes = 0
        old_codes = None
        flags = {
            'skip0': False,
            'skip1': False,
            'skip2': False,
            'skip3': False,
            'uncond1': False,
            'uncond2': False,
            'uncond3': False,
            'base': False,
        }

        # 设置对应标志为 True
        skip_value_map = {
            0: 'skip0',
            1: 'skip1',
            2: 'skip2',
            3: 'skip3',
        }
        cond_value_map = {
            0: 'base',
            1: 'uncond1',
            2: 'uncond2',
            3: 'uncond3',
        }
        cond = False

        for si, pn in enumerate(scale_schedule):  # si: i-th segment
            if flags['skip3'] and si == 10:
                break
            if flags['skip2'] and si == 11:
                break
            if flags['skip1'] and si == 12:
                break
            # print(f"\n[debug] --- current scale: {si} ({pn[1]}x{pn[2]}) ---")
            # print(f'{cond_BD_or_gss.shape=}')

            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si + 1)]), None)

            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if layer_idx == 0 and ((flags['uncond1'] and si >= 12) 
                                       or (flags['uncond2'] and si >= 11) 
                                       or (flags['uncond3'] and si >= 10)):
                    last_stage = last_stage[:B]
                    cond_BD_or_gss = cond_BD_step12; #print(f'{cond_BD_step12.shape=}')
                    ca_kv = ca_kv_step12
                    cond = True

                for m in b.module:
                    if cond:
                        last_stage = m.forward_cond(
                            x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None,
                            attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, 
                            scale_ind=si)
                    else:
                        last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv,
                                       attn_bias_or_two_vector=None,
                                       attn_fn=attn_fn, scale_schedule=scale_schedule,
                                       rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                        last_stage = cfg * last_stage[:B] + (1 - cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1
            if cond:
                if (cfg != 1) and add_cfg_on_logits:
                    cond_logits_BlV = self.get_logits(last_stage, cond_BD[:B]).mul(1 / tau_list[si])
                    logits_BlV = cfg * cond_logits_BlV + (1 - cfg) * cond_logits_BlV
                else:
                    logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])
            else:
                if (cfg != 1) and add_cfg_on_logits:
                    # print(f'add cfg on add_cfg_on_logits')
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1 / tau_list[si])
                    logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
                else:
                    logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1 / tau_list[si])

            if self.use_bit_label:
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng,
                                                                                 top_k=top_k or self.top_k,
                                                                                 top_p=top_p or self.top_p,
                                                                                 num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng,
                                                                                top_k=top_k or self.top_k,
                                                                                top_p=top_p or self.top_p,
                                                                                num_samples=1)[:, :, 0]
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)  # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify:  # unpatchify operation
                        idx_Bld = idx_Bld.permute(0, 3, 1, 2)  # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2)  # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0, 2, 3, 1)  # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1)  # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label')  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]

                def extract_high_freq_sobel(image):
                    sobel_x = torch.tensor(
                        [[[[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]]]], dtype=torch.float32
                    ).to(image.device)

                    sobel_y = sobel_x.permute(0, 1, 3, 2)  # Transpose to get Sobel Y
                    gray_image = image.mean(dim=1, keepdim=True)
                    grad_x = F.conv2d(gray_image, sobel_x, padding=1)
                    grad_y = F.conv2d(gray_image, sobel_y, padding=1)
                    high_freq = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # Calculate gradient magnitude
                    return high_freq

                def extract_high_freq_ratio_fft_circle(img, freq_ratio=0.25):
                    """
                    img: torch.Tensor, shape [B, C, H, W], values in [0, 1] or [0, 255]
                    freq_ratio: float, freq_ratio * min(H, W) / 2
                    Return: shape [B, 1]
                    """
                    gray = img.mean(dim=1, keepdim=True).float()  # [B, 1, H, W]

                    fft = torch.fft.fft2(gray)  # Fast Fourier Transform
                    fft_shift = torch.fft.fftshift(fft)  # Shift to center
                    amplitude = torch.abs(fft_shift)  # Amplitude spectrum

                    B, C, H, W = amplitude.shape
                    cy, cx = H // 2, W // 2
                    radius = int(freq_ratio * min(H, W) / 2)

                    # Create a central circle mask: low frequency is 0, high frequency is 1
                    y = torch.arange(H, device=img.device).view(1, -1, 1).expand(1, H, W)
                    x = torch.arange(W, device=img.device).view(1, 1, -1).expand(1, H, W)
                    dist = ((y - cy) ** 2 + (x - cx) ** 2).sqrt()

                    mask = (dist > radius).float().to(amplitude.device)  # shape [1, H, W]
                    # Only unsqueeze once, so that mask shape is [1, 1, H, W]
                    mask = mask.unsqueeze(0)
                    # If batch is greater than 1, expand mask to all batches
                    mask = mask.expand(B, 1, H, W)

                    high_freq_energy = (amplitude * mask).sum(dim=[2, 3])
                    total_energy = amplitude.sum(dim=[2, 3])
                    hf_ratio = high_freq_energy / (total_energy + 1e-6)  # Avoid division by zero

                    return hf_ratio  # Output shape should be [B, 1]

                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1],
                                                  mode=vae.quantizer.z_interplote_up)
                    if si == 9:
                        new_img = F.interpolate(summed_codes, size=vae_scale_schedule[4],
                                                mode=vae.quantizer.z_interplote_up)
                        old_img = F.interpolate(old_codes, size=vae_scale_schedule[4],
                                                mode=vae.quantizer.z_interplote_up)
                        old_img = vae.decode(old_img.squeeze(-3))

                        new_img = vae.decode(new_img.squeeze(-3))
                        new_img_hf = extract_high_freq_sobel(new_img)
                        old_img_hf = extract_high_freq_sobel(old_img)
                        new_ratio = extract_high_freq_ratio_fft_circle(new_img, freq_ratio=0.4)

                        diff_hf = F.l1_loss(new_img_hf, old_img_hf)
                        sample_input = [[diff_hf.item(), new_ratio.item()]]
                        # Step 1: Step Skipping Prediction
                        skip_label = self.skip_scaler.transform(sample_input)
                        skip_pred = self.skip_model.predict(skip_label)[0]
                        flags[skip_value_map[skip_pred]] = True

                        # print(f'{skip_pred=}')
                        cond_pred = None
                        if skip_pred == 0:
                            # Step 2: Uncondition Branch Replacement Prediciton
                            uncond_label = self.uncond_scaler.transform(sample_input)
                            uncond_pred = self.uncond_model.predict(uncond_label)[0]
                            flags[cond_value_map[uncond_pred]] = True
                        # print(f'{flags=}')
                    if si == 8:
                        old_codes = summed_codes.clone()

                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1],
                                               mode=vae.quantizer.z_interplote_up)  # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3)  # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify:  # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2)  # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1)  # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0, 2, 1])  # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()  # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si,
                                                                                               num_stages_minus_1 + 1,
                                                                                               accu_BChw, h_BChw,
                                                                                               scale_schedule)

            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs // B, 1, 1)

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []

        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img


if __name__ == '__main__':

    import argparse
    import time

    from torch import autocast

    from tools.run_infinity import load_transformer, load_visual_tokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_seed = random.randint(0, 10000)

    args=argparse.Namespace(
        pn='1M',
        model_path=None,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=None,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=None,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        save_file='tmp.jpg',
        # 
        enable_model_cache=0)

    # load vae
    vae = load_visual_tokenizer(args)
    # infinity = load_transformer(vae, args)
    # --> do not load transformer weight, for fast debug
    if args.model_type == 'infinity_2b':
        model_kwargs = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True), torch.no_grad():
        infinity_test: SkipVAR_Infinity = SkipVAR_Infinity(
            vae_local=vae, text_channels=2048, text_maxlen=512,
            shared_aln=True, raw_scale_schedule=None,
            checkpointing='full-block',
            customized_flash_attn=False,    # default: False
            fused_mlp=False,                # default: False
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=0,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            pn='1M',
            apply_spatial_patchify=0,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={args.bf16}')

        if args.bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        # *------ t2i forward ------*
        h_div_w = 1/1 # aspect ratio, height:width
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        # dummy_input
        text_cond_tuple = (
            torch.randn(size=[9, 2048], device=device, dtype=torch.float32),
            [9],
            torch.tensor([0, 9], device=device, dtype=torch.int32),
            9
        )

        # warmup GPU
        warmup_iterations = 10
        print(f"Starting GPU warm-up for {warmup_iterations} iterations...")
        with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
            for _ in range(warmup_iterations):
                _, _, _ = infinity_test.autoregressive_infer_cfg(
                    vae=vae,
                    scale_schedule=scale_schedule,
                    label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
                    B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                    cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                    returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                    cfg_exp_k=0.0, cfg_insertion_layer=[0],
                    vae_type=32, softmax_merge_topk=-1,
                    ret_img=True, trunk_scale=1000,
                    gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                    sampling_per_bits=1,
                )
                torch.cuda.synchronize(device=device)
        print("GPU warm-up finished.")

        # Influence speed test
        num_test_iterations = 50
        timings = []

        print(f"Starting inference speed test for {num_test_iterations} iterations...")
        with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
            stt = time.time()
            for i in range(num_test_iterations):
                start_time = time.perf_counter()    # for accurate timing

                _, _, img_list = infinity_test.autoregressive_infer_cfg(
                    vae=vae,
                    scale_schedule=scale_schedule,
                    label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
                    B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                    cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                    returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                    cfg_exp_k=0.0, cfg_insertion_layer=[0],
                    vae_type=32, softmax_merge_topk=-1,
                    ret_img=True, trunk_scale=1000,
                    gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                    sampling_per_bits=1,
                )
                torch.cuda.synchronize(device=device)   # *Important*: Ensure that all CUDA operations are completed before recording the time

                end_time = time.perf_counter()
                timings.append(end_time - start_time)
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}/{num_test_iterations} done.")

        print("Inference speed test finished.")
        
        batch_size = 1
        avg_latency = sum(timings) / len(timings)
        std_latency = torch.tensor(timings).std().item()
        throughput = batch_size / avg_latency if avg_latency > 0 else float('inf')

        print(f"\n--- Inference Performance ---")
        print(f"Batch Size: {batch_size}")
        print(f"Average Latency: {avg_latency * 1000:.2f} ms")
        print(f"Latency StdDev: {std_latency * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")

        # --- print first and last times to observe warmup effect ---
        # print("\nFirst 5 latencies (ms):")
        # for t in timings[:5]:
        #     print(f"{t*1000:.2f}")

        # print("\nLast 5 latencies (ms):")
        # for t in timings[-5:]:
        #     print(f"{t*1000:.2f}")
        
        img = img_list[0]
        print(f"{img.shape=}")
