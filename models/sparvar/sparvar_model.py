from contextlib import nullcontext
from functools import partial
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

from models.infinity.flex_attn import FlexAttn
from models.infinity.basic_infinity import (
    AdaLNBeforeHead, 
    SelfAttnBlock, 
    flash_attn_func, 
    flash_fused_op_installed, 
    FastRMSNorm, 
    precompute_rope2d_freqs_grid
)
from models.infinity.infinity_model import (
    MultiInpIdentity, 
    MultipleLayers, 
    SharedAdaLin, 
    TextAttentivePool, 
    sample_with_top_k_top_p_also_inplace_modifying_logits_
)
from models.sparvar.basic_sparvar import CrossAttnBlock

import utils.dist as dist
from utils.dist import for_visualize
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

try:
    from models.infinity.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


class SparVAR_Infinity(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=0, text_maxlen=0,     # text-cond generation
        selecting_idx=None,                 # class-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        customized_flash_attn=False, fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        cache_dir=None,          # debug, timm load_model has cache_dir default
        skip_last_scales: int = 0,
        drop_uncond_last_scales: int = 0,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales

        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.use_bit_label: {self.use_bit_label}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)
        
        customized_kernel_installed = any('Infinity' in arg_name for arg_name in flash_attn_func.__code__.co_varnames)
        # customized_kernel_installed = True    # for flex_attn test, false!
        self.customized_flash_attn = customized_flash_attn and customized_kernel_installed
        if customized_flash_attn and not customized_kernel_installed:
            import inspect, warnings
            file_path = inspect.getsourcefile(flash_attn_func)
            line_number = inspect.getsourcelines(flash_attn_func)[1]
            info = (
                f'>>>>>> Customized FlashAttention2 is not installed or compiled, but specified in args by --flash=1. Set customized_flash_attn = False. <<<<<<\n'
                f'>>>>>> `flash_attn_func` is in [line {line_number}] [file {file_path}] <<<<<<\n'
                f'>>>>>> {flash_attn_func.__code__.co_varnames=} <<<<<<\n'
            )
            warnings.warn(info, ImportWarning)
            print(info, flush=True)
        
        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = 1
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        # self.rng = torch.Generator(device='cuda')
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen
        self.t2i = text_channels != 0
        
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        if self.t2i:
            self.selecting_idx = None
            self.num_classes = 0
            self.D = self.C
            
            cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
            rng = torch.Generator(device='cpu')
            rng.manual_seed(0)
            torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
            cfg_uncond /= self.Ct5 ** 0.5
            if rand_uncond:
                self.register_buffer('cfg_uncond', cfg_uncond)
            else:
                self.cfg_uncond = nn.Parameter(cfg_uncond)
            
            self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
            self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.Ct5, self.D),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.D, self.D),
            )
        else:   # class-label cond
            if selecting_idx is None:
                num_classes = 1000
                print(f'======= WARNING: selecting_idx not specified, set to 1/{num_classes} @ {dist.get_device()} =======')
                selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
            self.selecting_idx = selecting_idx
            self.num_classes = selecting_idx.shape[-1]
            self.D = self.C
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        if self.rope2d_each_sa_layer:
            rope2d_freqs_grid = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw)
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                # fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                swiglu=swiglu, customized_flash_attn=self.customized_flash_attn, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))

        print(
            f'\n[constructor]  ==== customized_flash_attn={self.customized_flash_attn} (using_flash={sum((b.sa.using_flash if self.t2i else b.attn.using_flash) for b in self.unregistered_blocks)}/{self.depth}), fused_mlp={fused_mlp} (fused_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.unregistered_blocks)}/{self.depth}) ==== \n'
            f'    [Infinity config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}, swiglu={swiglu} num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}\n'
            f'    [drop ratios] drop_rate={drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})\n',
            f'    [skip scales] skip last {skip_last_scales} scales\n',
            f'    [drop uncond branch] drop last {drop_uncond_last_scales} scales\n',
            end='\n\n', flush=True
        )
        self.skip_last_scales = skip_last_scales
        self.drop_uncond_last_scales = drop_uncond_last_scales

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "var_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = False
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L,
                                        auto_padding=auto_padding)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0):
        ptr = 0
        x_BLC_list = []
        for scale_ind, patch_t_h_w in enumerate(scale_schedule):
            scale_seq_len = np.array(patch_t_h_w).prod()
            x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w, c]
            ptr += scale_seq_len
            x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule)
            x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.customized_flash_attn:
                Infinity_visible_kvlen = self.Infinity_visible_kvlen[:l_end]
                Infinity_invisible_qlen = self.Infinity_invisible_qlen[:l_end]
                attn_bias_or_two_vector = (Infinity_visible_kvlen, Infinity_invisible_qlen)
                # todo: solve need_to_pad here
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        # [2. block loop]
        SelfAttnBlock.forward, CrossAttnBlock.forward
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid)

        # [3. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,    # [(1, 1, 1), (1, 2, 2), (1, 4, 4), ..., (1, 48, 48), (1, 64, 64)]
        label_B_or_BLT=None,    # tuple(torch.Size([9, 2048]), list[9], tensor[0, 9] shape=[2])
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,    # top_k=900, top_p=0.97
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,               # returns_vemb=1
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],     # cfg_insertion_layer=[0], CFG to Transformer layer: 0 for logits; 1 for probability, 负值表示从末尾开始的第n层
        vae_type=0, softmax_merge_topk=-1, ret_img=False,   # vae_type=32, ret_img=True
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,   # True for infra, to start KV-Cache
        save_img_path=None,
        sampling_per_bits=1,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)     # for each scale has cfg and tau
        assert len(tau_list) >= len(scale_schedule)

        # --- for drop uncond branch ---
        if self.drop_uncond_last_scales > 0:
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
        if self.apply_spatial_patchify:     # default: 0
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        # ------ text prompt process: for CFG ------
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)      # [2*t_len, text_dim(2048)]
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)      # tensor[0, 9, 18], adjust the cumulative seq_len of 2*bs
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B
        # ------------------------------------------

        kv_compact = self.text_norm(kv_compact)     # FastRMSNorm
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))    # sos: [2, v_dim(2048)], cond_BD: [2, v_dim]
        kv_compact = self.text_proj_for_ca(kv_compact)                                      # kv_compact: [2*len, v_dim], linear -> tanh -> linear
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)      # [bs, 1, v_dim], Transformer first scale input

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()      # shared_ada_lin: silu -> SharedAdaLin; gss: gamma, scale, shift for AdaLN
        accu_BChw, cur_L, ret = None, 0, []     # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []      # Bl: 存储每个尺度预测的 token 索引的列表; Bld: 存储每个尺度预测的逐位 token 标签的列表

        if inference_mode:      # start Transformer block's attn KV-Cache
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
            # for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).update_batch(B)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)

        # ------ where and how to use CFG ------
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        # ---------------------------------------

        num_stages_minus_1 = len(scale_schedule) - 1
        summed_codes = 0
        # ------ auto-regressive scale ------             si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
        for si, pn in enumerate(scale_schedule):        # si: i-th segment, pn: current scale patch number
            # skip last scales
            if self.skip_last_scales == 2 and (48 == pn[2] or 64 == pn[2]):
                continue
            elif self.skip_last_scales == 1 and 64 == pn[2]:
                continue
            # print(f"\n[debug] --- current scale: {si} ({pn[1]}x{pn[2]}) ---")

            cfg = cfg_list[si]                          # get current scale si's CFG
            if si >= trunk_scale:                       # trunk_scale=1000
                break
            cur_L += np.array(pn).prod()                # current cumulative token length

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:      # test default = 0
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # ------ Transformer blocks ------
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                
                if layer_idx == 0 and ((self.drop_uncond_last_scales == 1 and si >= 12) 
                                       or (self.drop_uncond_last_scales == 2 and si >= 11)
                                       or (self.drop_uncond_last_scales == 3 and si >= 10)):
                    last_stage = last_stage[:B]
                    cond_BD_or_gss = cond_BD_step12 #print(f'{cond_BD_step12.shape=}')
                    ca_kv = ca_kv_step12
                
                for m in b.module:
                    last_stage = m(
                        x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, 
                        attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, 
                        scale_ind=si, layer_ind=layer_idx, 
                    )
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                        last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1
            # --------------------------------

            if (cfg != 1) and add_cfg_on_logits:
                # print(f'add cfg on add_cfg_on_logits')
                if last_stage.shape[0] * 2 == cond_BD.shape[0]:
                    logits_BlV = self.get_logits(last_stage, cond_BD[:B]).mul(1/tau_list[si])
                    logits_BlV = cfg * logits_BlV + (1-cfg) * logits_BlV
                else:
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])       # get 2*bs(cond and uncond)'s logits
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            # ------ sample from logits_BlV, get current scale predicted tokens ------
            if self.use_bit_label:      # use_bit_label = 1
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            # ------ process sampled tokens, prepare next AR step's input ------
            if vae_type != 0:           # vae_type = 32
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            # -------------------------------------------------------------------

            if si != num_stages_minus_1:    # not last scale
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        if inference_mode:  # disable attn's KV-Cache --> free memory
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        # ------ decode latent feature to img ------
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            sab: Union[SelfAttnBlock, CrossAttnBlock]
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                if self.t2i:
                    sab.sa.proj.weight.data.mul_(scale)
                    sab.ca.proj.weight.data.mul_(scale)
                else:
                    sab.attn.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError


if __name__ == '__main__':

    import argparse
    import cv2
    import os
    import time
    from torch import autocast
    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # seed = random.randint(0, 10000)
    seed = 10086
    enable_positive_prompt=0

    # model_type = 'infinity_2b'
    model_type = 'infinity_8b'
    load_weight = True
    latency_profile = True
    prompt = """A group of students in a class"""

    if '2b' in model_type:
        vae_type = 32
        apply_spatial_patchify = 0
        checkpoint_type = 'torch'
        model_path = "pretrained_models/infinity/Infinity/infinity_2b_reg.pth"
        vae_path = 'pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth'
        cfg = 3; tau = 0.5
        model_kwargs = dict(
            depth=32, embed_dim=2048, num_heads=2048//128, 
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    if '8b' in model_type:
        vae_type=14
        apply_spatial_patchify = 1
        checkpoint_type = 'torch_shard'
        model_path = "pretrained_models/infinity/Infinity/infinity_8b_weights"
        vae_path = 'pretrained_models/infinity/Infinity/infinity_vae_d56_f8_14_patchify.pth'
        cfg = 3; tau = 1.0
        model_kwargs = dict(
            depth=40, embed_dim=3584, num_heads=28, 
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    text_encoder_ckpt = 'pretrained_models/infinity/flan-t5-xl'
    
    if not load_weight:
        model_path = None
        vae_path = None
        text_encoder_ckpt = None

    args=argparse.Namespace(
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=vae_type,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type=model_type,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=apply_spatial_patchify,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type=checkpoint_type,
        seed=seed,
        bf16=1,
        save_file='tmp.jpg',
        # 
        enable_model_cache=0,
        # -- exp params ---
        attn_sink_scales=5,
        skip_last_scales=0,
        drop_uncond_last_scales=3)

    if not load_weight:
        # load vae
        vae = load_visual_tokenizer(args)
        # --> do not load transformer weight, for fast debug
        with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True), torch.no_grad():
            infinity_test: SparVAR_Infinity = SparVAR_Infinity(
                vae_local=vae, text_channels=2048, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing='full-block',
                customized_flash_attn=False,    # default: False
                fused_mlp=False,                 # default: False
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=0,                # default: 0
                add_lvl_embeding_only_first_block=1,
                use_bit_label=1,
                rope2d_each_sa_layer=1,
                rope2d_normalized_by_hw=2,
                pn='1M',
                apply_spatial_patchify=apply_spatial_patchify,
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
    else:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity_test = load_transformer(vae, args)
    
    # *------ t2i forward ------*
    h_div_w = 1/1 # aspect ratio, height:width
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    if not load_weight:
        # dummy_input
        text_cond_tuple = (
            torch.randn(size=[9, 2048], device=device, dtype=torch.float32),
            [9],
            torch.tensor([0, 9], device=device, dtype=torch.int32),
            9
        )

        if not latency_profile:
            _, _, img_list = infinity_test.autoregressive_infer_cfg(
                vae=vae,
                scale_schedule=scale_schedule,
                label_B_or_BLT=text_cond_tuple, g_seed=seed,
                B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                cfg_exp_k=0.0, cfg_insertion_layer=[0],
                vae_type=32, softmax_merge_topk=-1,
                ret_img=True, trunk_scale=1000,
                gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                sampling_per_bits=1,
            )
            img = img_list[0]
            print(f"{img.shape=}")
        
        else:
            # warmup GPU
            warmup_iterations = 10
            print(f"Starting GPU warm-up for {warmup_iterations} iterations...")
            with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                for _ in range(warmup_iterations):
                    _, _, _ = infinity_test.autoregressive_infer_cfg(
                        vae=vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=seed,
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
            num_test_iterations = 100
            timings = []

            print(f"Starting inference speed test for {num_test_iterations} iterations...")
            with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                stt = time.time()
                for i in range(num_test_iterations):
                    start_time = time.perf_counter()    # for accurate timing

                    _, _, img_list = infinity_test.autoregressive_infer_cfg(
                        vae=vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=seed,
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
    
    else:

        generated_image = gen_one_img(
            infinity_test,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=0,
        )
        args.save_file = f'{model_type}_test.jpg'
        os.makedirs(os.path.dirname(os.path.abspath(args.save_file)), exist_ok=True)
        cv2.imwrite(args.save_file, generated_image.cpu().numpy())
        print(f'Save to {os.path.abspath(args.save_file)}')
