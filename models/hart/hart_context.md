
HART 的论文和核心代码如下所示：

configuration.py HART model的相关配置

```python
from typing import Optional
from transformers import PretrainedConfig


__all__ = [
    "HARTAutoEncoderConfig", 
    "HARTAutoEncoderWithDiscConfig",
    "VARTransformerConfig",
    "VARTransformerT2IConfig",
    "HARTForC2IConfig",
    "HARTForT2IConfig",
]


class HARTAutoEncoderConfig(PretrainedConfig):
    model_type = "hart_autoencoder"

    def __init__(
        self,
        vocab_size=4096,
        z_channels=32,
        ch=160,
        dropout=0.0,
        beta=0.25,
        using_znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        default_qresi_counts=0,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        test_mode=False,
        ch_mult=(1, 1, 2, 2, 4),
        levels=[8, 8, 8, 6, 5],
        quantizer_type: str = "var",
        hybrid: bool = False,
        disable_quant_resi: bool = False,
        freeze_codebook_for_hybrid: bool = True,
        double_decoder=False,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.z_channels = z_channels
        self.ch = ch
        self.dropout = dropout
        self.beta = beta
        self.using_znorm = using_znorm
        self.quant_conv_ks = quant_conv_ks
        self.quant_resi = quant_resi
        self.share_quant_resi = share_quant_resi
        self.default_qresi_counts = default_qresi_counts
        self.v_patch_nums = v_patch_nums
        self.test_mode = test_mode
        self.ch_mult = ch_mult
        self.levels = levels
        self.quantizer_type = quantizer_type
        self.hybrid = hybrid
        self.disable_quant_resi = disable_quant_resi
        self.freeze_codebook_for_hybrid = freeze_codebook_for_hybrid
        self.double_decoder = double_decoder


class HARTAutoEncoderWithDiscConfig(HARTAutoEncoderConfig):
    model_type = "hart_autoencoder_with_disc"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class VARTransformerConfig(PretrainedConfig):
    model_type = "var_transformer"

    def __init__(
        self,
        vae_path: Optional[str] = None,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        mlp_type="gpt2",
        attn_type="gpt2",
        disable_aln=False,
        use_timestep_embed=False,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
        latent_condition_weight=1.0,
        **kwargs,
    ):
        super().__init__()

        self.vae_path = vae_path
        self.num_classes = num_classes
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_eps = norm_eps
        self.shared_aln = shared_aln
        self.cond_drop_rate = cond_drop_rate
        self.attn_l2_norm = attn_l2_norm
        self.patch_nums = patch_nums
        self.flash_if_available = flash_if_available
        self.fused_if_available = fused_if_available
        self.mlp_type = mlp_type
        self.attn_type = attn_type
        self.disable_aln = disable_aln
        self.use_timestep_embed = use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn
        self.diffusion_head_repeats = kwargs.get("diffusion_head_repeats", 1)
        self.latent_condition_weight = latent_condition_weight


class VARTransformerT2IConfig(PretrainedConfig):
    model_type = "var_transformer_t2i"

    def __init__(
        self,
        vae_path: Optional[str] = None,
        context_token=77,
        context_dim=768,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        mlp_type="gpt2",
        attn_type="gpt2",
        disable_aln=False,
        use_timestep_embed=False,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
        **kwargs,
    ):
        super().__init__()

        self.vae_path = vae_path
        self.context_token = context_token
        self.context_dim = context_dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_eps = norm_eps
        self.shared_aln = shared_aln
        self.cond_drop_rate = cond_drop_rate
        self.attn_l2_norm = attn_l2_norm
        self.patch_nums = patch_nums
        self.flash_if_available = flash_if_available
        self.fused_if_available = fused_if_available
        self.mlp_type = mlp_type
        self.attn_type = attn_type
        self.disable_aln = disable_aln
        self.use_timestep_embed = use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn


class HARTForC2IConfig(VARTransformerConfig):
    model_type = "hart_transformer_c2i"

    def __init__(
        self,
        diff_width=1024,
        diff_depth=6,
        num_sampling_steps="8",
        diffusion_batch_mul=4,
        sampler="iddpm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diff_width = diff_width
        self.diff_depth = diff_depth
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul
        self.sampler = sampler


class HARTForT2IConfig(VARTransformerT2IConfig):
    model_type = "hart_transformer_t2i"

    def __init__(
        self,
        diff_width=1024,
        diff_depth=6,
        num_sampling_steps="8",
        diffusion_batch_mul=4,
        sampler="iddpm",
        use_context_norm=False,
        context_norm_scale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diff_width = diff_width
        self.diff_depth = diff_depth
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul
        self.sampler = sampler
        self.diffusion_head_repeats = kwargs.get("diffusion_head_repeats", 1)
        self.use_context_norm = use_context_norm
        self.context_norm_scale = context_norm_scale
```


hart_transformer_t2i.py HART model的模型架构
```python
import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from models.diffusion.diffloss import DiffLoss
from models.hart.basic_hart import (AdaLNBeforeHead, AdaLNSelfAttn,
                                    LlamaRMSNormFused, TimestepEmbedder)
from models.hart.configuration import HARTForT2IConfig
from models.hart.hart_autoencoder import HARTAutoEncoder
from models.hart.hart_autoencoder_with_disc import HARTAutoEncoderWithDisc
from models.helpers import (_list_ckpt_files, _load_all_states,
                            _strip_majority_prefix, gumbel_softmax_with_rng,
                            sample_with_top_k_top_p_)
from models.vae.hart_hybrid_quant import HARTHybridQuantizer
from utils.dist import get_device


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda(),
    ).bool()
    return masking


class CopyableGenerator(torch.Generator):
    def __deepcopy__(self, memo):
        new_generator = CopyableGenerator(device=self.device)
        new_generator.set_state(self.get_state())
        return new_generator


class HARTForT2I(PreTrainedModel):
    config_class = HARTForT2IConfig

    def __init__(self, config: HARTForT2IConfig):
        super().__init__(config)
        self.supports_gradient_checkpointing = True

        # 0. hyperparameters
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        vae_path = config.vae_path

        if vae_path is None:
            vae_path = os.path.join(
                os.path.dirname(config._name_or_path.rstrip("/")), "tokenizer"
            )
        depth = config.depth
        drop_rate = config.drop_rate
        cond_drop_rate = config.cond_drop_rate
        drop_path_rate = config.drop_path_rate
        attn_drop_rate = config.attn_drop_rate
        mlp_ratio = config.mlp_ratio
        norm_eps = config.norm_eps
        shared_aln = config.shared_aln
        attn_l2_norm = config.attn_l2_norm
        context_token = config.context_token
        context_dim = config.context_dim
        patch_nums = config.patch_nums
        flash_if_available, fused_if_available = (
            config.flash_if_available,
            config.fused_if_available,
        )
        self.mlp_type = mlp_type = config.mlp_type
        self.attn_type = attn_type = config.attn_type
        if self.attn_type == "gpt2":
            norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        elif self.attn_type == "llama":
            norm_layer = partial(LlamaRMSNormFused, eps=norm_eps)
        else:
            raise NotImplementedError
        self.disable_aln = config.disable_aln
        self.use_timestep_embed = use_timestep_embed = config.use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode = config.sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn = config.use_cross_attn

        self.diffusion_head_repeats = diffusion_head_repeats = (
            config.diffusion_head_repeats)

        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        mask_ratio_min = 0.5
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25,
            0,
            loc=1.0,
            scale=0.25)

        # vae_local = HARTAutoEncoderWithDisc.from_pretrained(vae_path).vae
        # !====== to solve transformers (!=4.42.2) version mismatch ======
        # ---------- 1) 先拿到backbone ----------
        vae = HARTAutoEncoderWithDisc.from_pretrained(
            vae_path,
            # 新版 transformers 警告：用 dtype 替代 torch_dtype
            dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=False,   # 先关低内存懒加载
        ).vae

        # print("before to_empty, any_meta:", any(p.is_meta for p in vae.parameters()))

        # ---------- 2) 将 meta 模块物化到 CPU ----------
        # 这是 PyTorch 对“从 meta 移动设备”的推荐做法
        if any(p.is_meta for p in vae.parameters()):
            vae.to_empty(device='cpu')  # 关键一步：真正分配张量内存
        # print("after  to_empty, any_meta:", any(p.is_meta for p in vae.parameters()))

        # ---------- 3) 读取并合并所有权重文件 ----------
        files = _list_ckpt_files(vae_path)
        assert files, f"No checkpoint files under: {vae_path}"
        state = _load_all_states(files)
        assert isinstance(state, dict) and state, f"Empty/invalid state dict from {files}"

        # ---------- 4) 去掉多余前缀 'vae.' 以对齐子模块 ----------
        state = _strip_majority_prefix(state, prefix="vae.")

        # （如仍有统一的 'module.' 等前缀，可再调用一次）
        # state = _strip_majority_prefix(state, prefix="module.")

        # ---------- 5) 加载权重（不解包返回值；很多自定义实现返回 None） ----------
        vae.load_state_dict(state, strict=False)

        # 打印对齐情况（自己做集合差最稳）
        named = dict(vae.named_parameters())
        missing = [k for k in named.keys() if k not in state]
        unexpected = [k for k in state.keys() if k not in named]
        # print(f"[manual load] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:10]:    print("  sample missing:", missing[:10])
        if unexpected[:10]: print("  sample unexpected:", unexpected[:10])

        # print("after  manual load, any_meta:", any(p.is_meta for p in vae.parameters()))

        # ---------- 6) 再移动到 CUDA ----------
        vae = vae.to("cuda", non_blocking=True)
        vae.requires_grad_(False)
        print("after  .to(cuda), any_meta:", any(p.is_meta for p in vae.parameters()))

        vae_local = vae
        # !==================================================================

        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = tuple(patch_nums)
        self.L = sum(pn**2 for pn in self.patch_nums)
        # self.first_l = self.patch_nums[0] ** 2
        self.first_l = context_token
        self.begin_ends = []
        self.begin_ends.append((0, context_token))
        cur = context_token
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = CopyableGenerator(device=get_device())

        # 1. input (word) embedding
        quant: HARTHybridQuantizer = vae_local.quantize
        self.vae_proxy: Tuple[HARTAutoEncoder] = (vae_local,)
        self.vae_quant_proxy: Tuple[HARTHybridQuantizer] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.context_token = context_token
        self.context_dim = context_dim
        self.context_shape = (context_token, context_dim)
        self.context_embed = nn.Linear(context_dim, self.D)
        if config.use_context_norm:
            self.context_norm = norm_layer(context_dim, scale=config.context_norm_scale)
        else:
            self.context_norm = nn.Identity()
        
        nn.init.trunc_normal_(self.context_embed.weight.data, mean=0, std=init_std)
        if attn_type == "gpt2" or self.context_token == 0:
            # gpt2 uses absolute pos emb for context tokens
            # c2i also adds this absolute pos emb
            self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
            nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        else:
            self.pos_start = None
        
        # 3. absolute position embedding
        self.last_level_pns = self.patch_nums[-1] ** 2
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            if i > 0:
                pe = torch.empty(1, pn * pn, self.C)
            else:
                pe = torch.empty(1, context_token, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L + context_token - 1, self.C)
        if self.attn_type == "gpt2":
            self.pos_1LC = nn.Parameter(pos_1LC)
        elif self.attn_type == "llama":
            self.pos_1LC = None
        else:
            raise NotImplementedError
        
        if not self.use_timestep_embed:
            # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        else:
            self.lvl_embed = TimestepEmbedder(embed_dim)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Identity()

        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                    mlp_type=mlp_type,
                    attn_type=attn_type,
                    max_position_embeddings=2
                    ** int(math.ceil(math.log2(self.L + context_token - 1))),
                    patch_nums=self.patch_nums,
                    context_token=self.context_token,
                    disable_aln=self.disable_aln,
                    sep_aln_pooling_mode=self.sep_aln_pooling_mode,
                    use_cross_attn=self.use_cross_attn,
                )
                for block_idx in range(depth)
            ]
        )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + context_token - 1, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L + context_token - 1, self.L + context_token - 1
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )
        print(attn_bias_for_masking.shape)

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.decoder_norm = norm_layer(self.C)
        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.last_level_pns, self.C))

        self.diffloss = DiffLoss(
            target_channels=self.Cvae,
            z_channels=self.C,
            width=config.diff_width,
            depth=config.diff_depth,
            num_sampling_steps=config.num_sampling_steps,
            sampler=config.sampler,
        )
        self.diffusion_batch_mul = config.diffusion_batch_mul
    
    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)

        return loss

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        final_stage=0,
        num_maskgit_iters=1,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # num_maskgit_iters = 1
        # final_stage = 2
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        sos = cond_BD = self.context_embed(     # [2*bs, 300, 1536]
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)))
        # Haotian: need to handle CFG here so we replicate context position ids
        context_position_ids = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)), dim=0,)

        b = context_mask.shape[0]
        context_mask = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0)))
        context_mask[b:, 0] = 1

        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)

        if self.pos_start is not None:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1) + lvl_pos[:, : self.first_l]
            )

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])   # [bs, 32, 64, 64]

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums[:-1]):  # si: i-th segment (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64)
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            if si > 0:
                cur_L += pn * pn
            else:
                cur_L += self.context_token
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                # Haotian: si used for position embed
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,)
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            # Haotian: Added for text-conditioned generation
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                2, 1, 1
            )  # double the batch sizes due to CFG

        ################ last stage maskgit ################
        si = len(self.patch_nums) - 1
        mask = torch.ones(B, self.last_level_pns).cuda()
        tokens = torch.zeros(B, self.last_level_pns, self.Cvae).cuda()
        orders = self.sample_orders(B)

        num_iter = num_maskgit_iters
        indices = list(range(num_iter))
        # generate latents with maskgit
        for step in indices:
            # mask_ratio = 1 - (step + 1) / num_iter
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.last_level_pns * mask_ratio)]).cuda()
            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, B, self.last_level_pns)
            if step >= num_iter - 1:
                mask_to_pred = mask[:B].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:B].bool(), mask_next.bool())
            mask = mask_next
            cur_mask = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            cur_mask = cur_mask.nonzero(as_tuple=True)
            x = next_token_map[cur_mask].reshape(2 * B, -1, self.C)
            for b in self.blocks:
                # Haotian: si used for position embed
                # note: m_maskgit makes sure that PEs are correct.
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=len(self.patch_nums) - 1,
                    m_maskgit=cur_mask,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            last_layer_cond = x
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            si = len(self.patch_nums) - 1
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            if final_stage == 0:
                # sample with diffusion model
                last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
                last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
                last_stage_discrete_cond = torch.cat(
                    [last_stage_discrete_cond, last_stage_discrete_cond], dim=0
                )
                last_stage_cond = self.decoder_norm(
                    last_layer_cond + last_stage_discrete_cond
                )
                bs, cur_seq_len, _ = last_stage_cond.shape
                ##### begin baseline sampling #####
                last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
                h_BChw_diff = self.diffloss.sample(
                    z=last_stage_cond, temperature=1.0, cfg=t
                )
                ##### end baseline sampling #####
                h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
                # [B, L, Cvae]
                h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
                # update feature map
                tokens[mask_to_pred] = (h_BChw + h_BChw_diff).reshape(-1, self.Cvae)
            else:
                tokens[mask_to_pred] = h_BChw.reshape(-1, self.Cvae)
        h_BChw_final = tokens.transpose(1, 2).reshape(
            B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
        )
        f_hat += h_BChw_final

        ################ last stage maskgit ################

        for b in self.blocks:
            b.attn.kv_caching(False)
        return (
            self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        )  # de-normalize, from [-1, 1] to [0, 1]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.last_level_pns)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        
        return orders
    
    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        # we cannot mask out all the tokens
        num_masked_tokens = min(int(np.ceil(seq_len * mask_rate)), seq_len - 32)
        mask = torch.zeros(bsz, seq_len, device=x.device)
        # all first few stages are kept
        mask_keep = torch.zeros(
            bsz, self.L - seq_len + self.context_token - 1, device=x.device
        )
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        mask_full = torch.cat([mask_keep, mask], dim=1).contiguous()
        
        return mask_full, mask
    
    def forward(
        self,
        context: torch.Tensor,
        x_BLCv_wo_first_l: torch.Tensor,
        context_position_ids: torch.Tensor,
        context_mask: torch.Tensor,
        last_layer_gt: torch.Tensor = None,
        last_layer_gt_discrete: torch.Tensor = None,
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = (
            self.begin_ends[self.prog_si]
            if self.prog_si >= 0
            else (0, self.L + self.context_token - 1)
        )
        B = x_BLCv_wo_first_l.shape[0]
        orders = self.sample_orders(bsz=B)
        mask, mask_wo_prev_stages = self.random_masking(
            x_BLCv_wo_first_l[:, -self.last_level_pns :, :], orders)
        
        mask_for_attn = (1 - mask)[:, self.context_token :].nonzero(as_tuple=True)
        mask = (1 - mask).nonzero(as_tuple=True)
        mask_wo_prev_stages = (1 - mask_wo_prev_stages).nonzero(as_tuple=True)
        last_layer_gt = last_layer_gt[mask_wo_prev_stages].reshape(
            B, -1, last_layer_gt.shape[-1])
        
        last_layer_gt_discrete = last_layer_gt_discrete[mask_wo_prev_stages].reshape(
            B, -1)
        
        ed = (
            last_layer_gt.shape[1]
            + self.L
            + self.context_token
            - 1
            - self.last_level_pns)

        with torch.cuda.amp.autocast(enabled=False):
            drop_pos = torch.where(
                torch.randn(B, device=context.device) < self.cond_drop_rate
            )[0]
            context[drop_pos] *= 0

            sos = cond_BD = self.context_embed(self.context_norm(context))
            if self.pos_start is not None:
                sos = sos.expand(B, self.first_l, -1) + self.pos_start.expand(
                    B, self.first_l, -1
                )
            else:
                sos = sos.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat(
                    (sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1
                )

            # apply maskgit
            x_BLC = x_BLC[mask].reshape(B, -1, x_BLC.shape[-1])

            if self.pos_1LC is not None:
                x_BLC += (
                    self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
                    + self.pos_1LC[:, :ed]
                )  # lvl: BLC;  pos: 1LC
            else:
                x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x_BLC = self._gradient_checkpointing_func(
                    b.forward_function,
                    x_BLC,
                    cond_BD_or_gss,
                    attn_bias,
                    mask_for_attn,
                    context_position_ids,
                    context_mask,
                )
            else:
                x_BLC = b(
                    x=x_BLC,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=attn_bias,
                    m_maskgit=mask_for_attn,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
        # parallel generation of discrete and continuous tokens
        x_BLC_logits, last_layer_cond = (
            x_BLC,
            x_BLC[:, self.L + self.context_token - 1 - self.last_level_pns :, :],
        )

        x_BLC_logits = self.get_logits(x_BLC_logits.float(), cond_BD)
        with torch.no_grad():
            # important to clone the last stage logits
            # Haotian: autoregressive LLM sometimes have this error:
            # RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
            try:
                idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits[
                        :, self.L + self.context_token - 1 - self.last_level_pns :
                    ]
                    .clone()
                    .detach(),
                    rng=self.rng,
                    top_k=600,
                    top_p=0.96,
                    num_samples=1,
                )[:, :, 0]
            except:
                idx_BL_sampled = last_layer_gt_discrete
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_stage_discrete_cond = self.word_embed(last_stage_discrete_embed)
        last_layer_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)

        last_layer_gt_continuous = last_layer_gt - last_stage_discrete_embed
        diff_loss = self.forward_diff_loss(
            z=last_layer_cond, target=last_layer_gt_continuous
        )
        # Haotian: important, we should start from self.context_token - 1.
        return (
            x_BLC_logits[:, self.context_token - 1 :, :],
            diff_loss,
            mask_wo_prev_stages,
        )  # logits BLV, V is vocab_size
    
    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
        conv_std_or_gain=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_nm.ada_lin[-1], "bias")
                and self.head_nm.ada_lin[-1].bias is not None
            ):
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fc2"):
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, "ada_lin"):
                sab.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if (
                    hasattr(sab.ada_lin[-1], "bias")
                    and sab.ada_lin[-1].bias is not None
                ):
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, "ada_gss"):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

        self.diffloss.initialize_weights()

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"
    

AutoConfig.register("hart_transformer_t2i", HARTForT2IConfig)
AutoModel.register(HARTForT2IConfig, HARTForT2I)
```

basic_hart.py HARTattention部分

```python
import functools
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import hart_backend.fused_kernels   # cd models/hart/kernels && python setup.py install or bash install.sh

from models.helpers import DropPath


# this file only provides the 3 blocks used in VAR transformer
__all__ = [
    "FFN",
    "TimestepEmbedder",
    "LlamaRMSNorm",
    "LlamaRMSNormFused",
    "LlamaMLP",
    "AdaLNSelfAttn",
    "AdaLNBeforeHead",
]


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None
try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    pass
# automatically import faster attention implementations
try:
    import xformers
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass
try:
    from flash_attn import flash_attn_func  # qkv: BLHc, ret: BLHcq
except ImportError:
    pass
try:
    from torch.nn.functional import (
        scaled_dot_product_attention as slow_attn,  # q, k, v: BHLc
    )
except ImportError:

    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None:
            attn.add_(attn_mask)
        return (
            F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
            if dropout_p > 0
            else attn.softmax(dim=-1)
        ) @ value


class LlamaRMSNormFused(nn.Module):
    # Shang: kwargs for elementwise_affine
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, use_quant: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_quant = use_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (
            torch.empty_like(x, dtype=torch.int8)
            if self.use_quant
            else torch.empty_like(x)
        )
        self.weight.data = self.weight.data.to(x)
        hart_backend.fused_kernels.rms_norm(
            out, x, self.weight.data, self.variance_epsilon, self.use_quant
        )
        return out


# From Junsong and Enze's EfficientDiT codebase.
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        flag = False
        if len(t.shape) == 2:
            flag = True
            t = t[0]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.dtype
        )
        t_emb = self.mlp(t_freq)
        if not flag:
            return t_emb
        else:
            return t_emb.unsqueeze(0)

    @property
    def dtype(self):
        # return the data type of this model
        return next(self.parameters()).dtype


class SelfAttention(nn.Module):
    def __init__(
        self,
        block_idx,
        embed_dim=768,
        num_heads=12,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_l2_norm=False,
        flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(
            torch.zeros(embed_dim)
        )
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = (
            False  # flash_if_available and memory_efficient_attention is not None
        )

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self, x, attn_bias, si=-1, context_position_ids=None, context_mask=None
    ):
        B, L, C = x.shape

        qkv = F.linear(
            input=x,
            weight=self.mat_qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=(
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
                ),
                p=dropout_p,
                scale=self.scale,
            ).view(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )

        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 4]
        # inv_freq_expanded: [bs, head_size // 4, 1, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None, None]
            .float()
            .expand(position_ids.shape[0], -1, 1, 1)
            .repeat(1, 1, 1, 2)
        )
        # position_ids_expanded: [bs, 1, seq_len, 2]
        position_ids_expanded = position_ids[:, None, :].float()
        inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
        position_ids_expanded = position_ids_expanded.permute(0, 3, 1, 2).contiguous()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, 2, seq_len, head_size // 4]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, 2, seq_len, head_size // 2]
            cos = emb.cos()
            sin = emb.sin()
            # [bs, seq_len, 2, head_size // 2]
            cos = cos.transpose(2, 1).contiguous()
            sin = sin.transpose(2, 1).contiguous()
            cos = cos.reshape(cos.size(0), cos.size(1), -1)
            sin = sin.reshape(sin.size(0), sin.size(1), -1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FusedRoPEFuncWithPos(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,  # [B, S, D]
        tensor_format: str = "sbhd",
        # cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = hart_backend.fused_kernels.fused_rope_with_pos_forward_func(
                t, freqs, False
            )
        elif tensor_format == "bshd":
            output = hart_backend.fused_kernels.fused_rope_with_pos_forward_func(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        raise NotImplementedError("Not implemented yet")
        # freqs, = ctx.saved_tensors
        # if ctx.tensor_format == "sbhd":
        #     grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output, freqs, False)
        # elif ctx.tensor_format == "bshd":
        #     grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output.transpose(0, 1), freqs, True).transpose(0, 1)
        # else:
        #     raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        # return grad_input, None, None


class FusedLlamaRotaryEmbedding2DWithPos(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Haotian: now we have two directions x and y so inv_freq has a stride 4

        # NOTE: Shang: freq stride is 4 rather than 2. While freq is normalized by dim.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, seq_len=None, position_ids=None, tensor_format="bshd"):
        if position_ids is not None:
            inv_freq_expanded = (
                self.inv_freq[None, :, None, None]
                .float()
                .expand(position_ids.shape[0], -1, 1, 1)
                .repeat(1, 1, 1, 2)
            )
            # position_ids_expanded: [bs, 1, seq_len, 2]
            position_ids_expanded = position_ids[:, None, :].float()
            inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
            position_ids_expanded = position_ids_expanded.permute(
                0, 3, 1, 2
            ).contiguous()

            device_type = x.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                # freqs: [bs, 2, seq_len, head_size // 4]
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(2, 3)
                embs = torch.cat((freqs, freqs), dim=-1)

                embs = embs.transpose(2, 1).contiguous()
                embs = embs.reshape(embs.size(0), embs.size(1), -1)

            return FusedRoPEFuncWithPos.apply(x, embs, tensor_format)

        else:  # Original impl
            raise NotImplementedError("Not implemented yet")
            self.embs = self.embs.to(x.device)
            return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding1D(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 2]
        # inv_freq_expanded: [bs, head_size // 2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded: [bs, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, seq_len, head_size // 2]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, seq_len, head_size]
            cos = emb.cos()
            sin = emb.sin()
        # [bs, seq_len, head_size]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        # cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = hart_backend.fused_kernels.fused_rope_forward_func(t, freqs, False)
        elif tensor_format == "bshd":
            output = hart_backend.fused_kernels.fused_rope_forward_func(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (freqs,) = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = hart_backend.fused_kernels.fused_rope_backward_func(
                grad_output, freqs, False
            )
        elif ctx.tensor_format == "bshd":
            grad_input = hart_backend.fused_kernels.fused_rope_backward_func(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None


class FusedLlamaRotaryEmbedding1DWithPos(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        seq = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)

        self.embs = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None, position_ids=None, tensor_format="bshd"):
        if position_ids is not None:
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            # print(self.embs.shape)
            # print(context_position_ids.shape)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            embs = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
            return FusedRoPEFuncWithPos.apply(x, embs, tensor_format)
        else:  # Original impl
            self.embs = self.embs.to(x.device)
            return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


@functools.cache    # need python 3.9+
def get_position_ids(batch_size, patch_nums, device, si=-1, m_maskgit=None):
    # [batch_size, L]
    all_position_ids = []
    largest_patch_num = patch_nums[-1]
    if si == -1:
        pns = patch_nums
    else:
        pns = patch_nums[si : si + 1]
    for level_idx in range(len(pns)):
        patch_num = pns[level_idx]
        _x = torch.arange(patch_num, device=device)
        _y = torch.arange(patch_num, device=device)
        # [pn, pn, 2]
        cartesian = torch.stack(torch.meshgrid(_x, _y, indexing="ij"), dim=-1)
        # normalize to the size in the largest feature map
        coords = cartesian / patch_num * largest_patch_num
        # [pn * pn, 2]
        coords = coords.reshape(-1, 2)
        all_position_ids.append(coords)
    # [batch_size, L, 2]
    pos_ids = torch.cat(all_position_ids, 0).unsqueeze(0).repeat(batch_size, 1, 1)
    if m_maskgit is None:
        return pos_ids
    pos_ids = pos_ids[m_maskgit]
    return pos_ids.reshape(batch_size, -1, pos_ids.shape[-1])


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# unsqueeze_dim=2 because by default our qk has shape [batch_size, seq_len, heads, head_dim]
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    def __init__(
        self,
        block_idx,
        patch_nums,
        embed_dim=768,
        num_heads=12,
        attn_drop=0.0,
        proj_drop=0.0,
        max_position_embeddings=4096,
        rope_theta=10000,
        flash_if_available=True,
        attn_l2_norm=False,
        context_token=0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert patch_nums is not None
        self.context_token = context_token
        self.patch_nums = patch_nums
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )  # =64

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attn_l2_norm = False

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.rotary_emb_fused_with_pos = FusedLlamaRotaryEmbedding2DWithPos(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        if context_token != 0:
            self.context_rotary_emb = LlamaRotaryEmbedding1D(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            self.context_rotary_emb_fused_with_pos = FusedLlamaRotaryEmbedding1DWithPos(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(
            torch.zeros(embed_dim)
        )
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = (
            False  # flash_if_available and memory_efficient_attention is not None
        )

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    # @get_local('attn')
    def forward(
        self,
        x,
        attn_bias,
        si=-1,
        context_position_ids=None,
        context_mask=None,
        m_maskgit=None,
    ):
        B, L, C = x.shape
        # [B, L, 2]
        if self.context_token == 0:     # context_token == 300
            position_ids = get_position_ids(
                B, self.patch_nums, x.device, si=si, m_maskgit=m_maskgit
            )
        else:
            # *text to image
            # *level 0 does not appear in the position_ids
            # since it is included in context tokens
            # should be 679 tokens for 16x16 latent w/ default 10-stage VAR
            if si == -1:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(
                    -1
                ).unsqueeze(-1)
            elif si > 0:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si - 1, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(
                    -1
                ).unsqueeze(-1)
        # [B, context, 2]
        # if self.context_token > 0 and si <= 0:
        #     context_position_ids = get_position_ids_1d(B, self.context_token, x.device)

        qkv = F.linear(
            input=x,
            weight=self.qkv_proj.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype       # torch.float16
        # qkv: BL3Hc

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
            dim_unsqueeze = 2
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc
            dim_unsqueeze = 1

        if self.attn_l2_norm:       # False
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        ################## Use naive rotary embedding ##################
        # apply position embedding to visual tokens
        if self.context_token == 0:     # usually > 0
            # position_ids exist for c2i
            # or t2i when stage id != 0
            # or t2i training phase (stage id = -1)
            cos, sin = self.rotary_emb(v, position_ids)
        elif self.context_token > 0:
            if si == -1:
                # training, all tokens
                cos, sin = self.rotary_emb(v, position_ids)
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = torch.cat([cos_c, cos], 1), torch.cat([sin_c, sin], 1)
            elif si == 0:
                # inference step 1, only context tokens
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = cos_c, sin_c
            else:
                # si > 0, no need to add rotary emb for context
                # inference step > 1, only new tokens
                cos, sin = self.rotary_emb(v, position_ids)
        else:
            print("Context token cannot be negative", self.context_token)
            raise NotImplementedError
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze)
        ################## Use naive rotary embedding ##################

        ################## Use optimized rotary embedding ##################
        # if self.context_token == 0:
        #     # position_ids exist for c2i
        #     # or t2i when stage id != 0
        #     # or t2i training phase (stage id = -1)
        #     cos, sin = self.rotary_emb(v, position_ids)
        #     q, k = apply_rotary_pos_emb(
        #         q,
        #         k,
        #         cos,
        #         sin,
        #         unsqueeze_dim=dim_unsqueeze
        #     )
        # elif self.context_token > 0:
        #     if si == -1:
        #         # training, all tokens
        #         cos, sin = self.rotary_emb(v, position_ids)
        #         cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
        #         cos, sin = torch.cat([cos_c, cos], 1), torch.cat([sin_c, sin], 1)
        #         q, k = apply_rotary_pos_emb(
        #             q,
        #             k,
        #             cos,
        #             sin,
        #             unsqueeze_dim=dim_unsqueeze
        #         )
        #     elif si == 0:
        #         # inference step 1, only context tokens
        #         # NOTE: This branch for prompt
        #         # cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
        #         # cos, sin = cos_c, sin_c
        #         q = self.context_rotary_emb_fused_with_pos(q, position_ids=context_position_ids, tensor_format="bshd")
        #         k = self.context_rotary_emb_fused_with_pos(k, position_ids=context_position_ids, tensor_format="bshd")
        #     else:
        #         # NOTE: This branch for multi-scale generation
        #         # si > 0, no need to add rotary emb for context
        #         # inference step > 1, only new tokens
        #         # cos, sin = self.rotary_emb(v, position_ids)
        #         q = self.rotary_emb_fused_with_pos(q, position_ids=position_ids, tensor_format="bshd")
        #         k = self.rotary_emb_fused_with_pos(k, position_ids=position_ids, tensor_format="bshd")
        # else:
        #     print("Context token cannot be negative", self.context_token)
        #     raise NotImplementedError
        ################## Use optimized rotary embedding ##################

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        # print(f'[scale-{si}_layer-{layer_ind}] self-attn caching: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')
        print(f'[scale-{si}] self-attn caching: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=(
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
                ),
                p=dropout_p,
                scale=self.scale,
            ).view(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
        act_func="gelu",
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if act_func == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif act_func == "silu":
            self.act = nn.SiLU()
        else:
            raise NotImplementedError
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class LlamaMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.gate_proj = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.up_proj = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.down_proj = nn.Linear(self.hidden_features, self.out_features, bias=False)
        self.act_fn = nn.SiLU()
        self.fused_mlp_func = None

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            raise NotImplementedError
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def context_pooling(context_tokens, context_mask=None, mode="avg"):
    # context_tokens: [batch, context_tokens, embed_dim]
    # context_mask: [batch, context_tokens]
    if len(context_tokens.shape) == 2:
        # C2I
        return context_tokens
    assert len(context_tokens.shape) == 3 and context_tokens.shape[1] > 1
    if mode == "avg":
        c_mask = context_mask.unsqueeze(-1)
        # [batch, context_tokens, embed_dim]
        condition = context_tokens * c_mask.to(context_tokens.dtype)
        # [batch, 1, embed_dim] => averaging
        condition = condition / c_mask.sum(1).clamp_(1).unsqueeze(1)
        # [batch, 1, embed_dim]
        condition = condition.sum(1)
    elif mode == "max":
        # [batch, 1, embed_dim]
        condition = context_tokens.max(1, keepdims=False).values
    else:
        raise NotImplementedError
    return condition


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self,
        block_idx,
        last_drop_p,
        embed_dim,
        cond_dim,
        shared_aln: bool,
        norm_layer,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_l2_norm=False,
        flash_if_available=False,
        fused_if_available=True,
        mlp_type="gpt2",
        attn_type="gpt2",
        gpt2_mlp_act_func="gelu",
        max_position_embeddings=4096,
        patch_nums=None,
        context_token=0,
        disable_aln=False,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
    ):
        super().__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.disable_aln = disable_aln
        self.sep_aln_pooling_mode = sep_aln_pooling_mode

        if attn_type == "gpt2":
            self.attn = SelfAttention(
                block_idx=block_idx,
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available,)
        else:
            self.attn = LlamaAttention(
                block_idx=block_idx,
                patch_nums=patch_nums,
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                max_position_embeddings=max_position_embeddings,
                rope_theta=10000,
                proj_drop=drop,
                flash_if_available=flash_if_available,
                context_token=context_token,
                attn_l2_norm=attn_l2_norm,)
        if mlp_type == "gpt2":
            self.ffn = FFN(
                in_features=embed_dim,
                hidden_features=round(embed_dim * mlp_ratio),
                drop=drop,
                fused_if_available=fused_if_available,
                act_func=gpt2_mlp_act_func,)
        elif mlp_type == "llama":
            # MLP ratio = 4: mul 8 / 3
            self.ffn = LlamaMLP(
                in_features=embed_dim,
                hidden_features=int((embed_dim * mlp_ratio * 2) / 3 + 255) // 256 * 256,
                out_features=embed_dim,
                drop=drop,
                fused_if_available=fused_if_available,)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if not self.disable_aln:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        else:
            if self.shared_aln:
                self.scale_shift_table = nn.Parameter(
                    torch.randn(6, embed_dim) / embed_dim**0.5)
        self.fused_add_norm_fn = None
        self.use_cross_attn = use_cross_attn

        if self.use_cross_attn:
            self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)
        else:
            self.cross_attn = None
    
    def forward_function(
        self,
        x_BLC,
        cond_BD_or_gss,
        attn_bias,
        mask,
        context_position_ids=None,
        context_mask=None,
    ):
        return self(
            x=x_BLC,
            cond_BD=cond_BD_or_gss,
            attn_bias=attn_bias,
            m_maskgit=mask,
            context_position_ids=context_position_ids,
            context_mask=context_mask,)
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        si=-1,
        context_position_ids=None,
        context_mask=None,
        m_maskgit=None,
    ):  # C: embed_dim, D: cond_dim
        # We achieve multi-token conditioning through LLM attention mask.
        if not self.disable_aln:    # always False
            # if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            #     cond_BD = cond_BD.max(1, keepdims=True).values
            condition = context_pooling(
                cond_BD, context_mask=context_mask, mode=self.sep_aln_pooling_mode
            ).unsqueeze(1)

            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_lin(condition)
                .view(-1, 1, 6, self.C)
                .unbind(2)
            )
            x = x + self.drop_path(
                self.attn(
                    self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                    attn_bias=attn_bias,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    si=si,
                    m_maskgit=m_maskgit,
                ).mul_(gamma1)
            )
            if self.use_cross_attn:
                # xattn_mask = get_xattn_mask(context_mask)
                x[:, cond_BD.size(1) :] += self.cross_attn(
                    x[:, cond_BD.size(1) :],
                    cond_BD,
                    None
                )
            x = x + self.drop_path(
                self.ffn(
                    self.ln_wo_grad(x)
                    .mul(scale2.add(1))
                    .add_(shift2)
                ).mul(gamma2)
            )  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        else:
            if not self.shared_aln:
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x),
                        attn_bias=attn_bias,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        si=si,
                        m_maskgit=m_maskgit,
                    )
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :],
                        cond_BD,
                        None
                    )
                x = x + self.drop_path(self.ffn(self.ln_wo_grad(x)))
            else:
                # cond_BD: [batch, 1, embed_dim]
                condition = context_pooling(cond_BD, context_mask, mode="avg")
                # [batch, 6, embed_dim]
                adaln_modulator = self.scale_shift_table[None] + condition.unsqueeze(1)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = adaln_modulator.chunk(
                    6, dim=1
                )
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                        attn_bias=attn_bias,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        si=si,
                        m_maskgit=m_maskgit,
                    ).mul_(gamma1)
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :],
                        cond_BD,
                        None
                    )
                x = x + self.drop_path(
                    self.ffn(
                        self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
                    ).mul(gamma2)
                )
        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        # We achieve multi-token conditioning through LLM attention mask.
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True).values

        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)

```

请你基于上述代码，将SkipVAR的加速策略添加到HART中，给出具体要修改哪些代码，在修改过程中要注意哪些问题
