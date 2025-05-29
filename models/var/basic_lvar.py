import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.processing_utils import Unpack

from models.helpers import DropPath
from models.kernels.gla.chunk import chunk_gla
from models.kernels.gla.fused_chunk import fused_chunk_gla
from models.kernels.gla.fused_recurrent import fused_recurrent_gla
from models.kernels.layers_utils import get_unpad_data, index_first_axis, pad_input
from models.kernels.modules.activations import ACT2FN
from models.kernels.modules.convolution import ShortConvolution
from models.kernels.modules.fused_norm_gate import FusedRMSNormGated
from models.kernels.modules.layernorm import RMSNorm
from models.kernels.utils import Cache

from tools.visual_attn import VisualAttnMap

# if TYPE_CHECKING:
    # from transformers.processing_utils import Unpack
    


# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass

# automatically import faster attention implementations
try: 
    from xformers.ops import memory_efficient_attention
except ImportError: pass

try: 
    from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass

try: 
    from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: 
            attn.add_(attn_mask)
        
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        # --- for visual attn map test ---
        # self.using_flash = False
        # self.using_xform = False
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias,
                vis_attn_map: VisualAttnMap):
        B, L, C = x.shape
        
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, 
                       bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)   # [bs, L, 3, num_heads, head_dim]
        main_type = qkv.dtype       # train: torch.float16
        # qkv: BL3Hc
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        # if L == 256 and (vis_attn_map is not None):
        #     print("last scale: x.shape = {}".format(x.shape))
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1     # q or k or v: BLHc
        else: 
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2     # q or k or v: BHLc
        
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
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)      # influence: [bs, 5, 16, 64], 5 for scale=3 --> k: [bs, 16, 16, 64]
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type), 
                k.to(dtype=main_type), 
                v.to(dtype=main_type), 
                dropout_p=dropout_p, 
                softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type), 
                k.to(dtype=main_type), 
                v.to(dtype=main_type), 
                attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), 
                p=dropout_p, 
                scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p,
                            vis_attn_map=vis_attn_map).transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class GatedLinearAttention(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,      # if LayerNorm/RMSNorm uses learnable affine transformation parameters
        norm_eps: float = 1e-5,                         # LayerNorm/RMSNorm epsilon, to /eps not /0
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads    # for GQA
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)        # for each KV group
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormGated(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps
            )
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(
                hidden_size=self.head_v_dim,
                elementwise_affine=elementwise_affine,
                eps=norm_eps
            )
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        hidden_states: torch.Tensor,                        # [bs, seq_len, ]
        attention_mask: Optional[torch.Tensor] = None,      # for padding, [bs, seq_len]
        past_key_values: Optional[Cache] = None,            # for AR decode
        use_cache: Optional[bool] = False,                  # influence is True
        output_attentions: Optional[bool] = False,          # if output attn weight (GLA not)
        **kwargs: Unpack[Dict]                              # such as cu_seqlens
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]    # [bs, D, D]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k, gk = (repeat(x, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_k_dim) for x in (k, gk))
            v = repeat(v, '... (h d) -> ... (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k, gk = (rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim) for x in (k, gk))
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        # recurrent_state = last_state if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, '... (h d) -> ... h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, '... h d -> ... (h d)')
            else:
                o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), '... h d -> ... (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()

        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias,
                vis_attn_map: VisualAttnMap = None):   # C: embed_dim, D: cond_dim     x: train[bs, 680, 1024], test[16, 1, 1024]
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)     # cond_BD: [bs, 1024] --ada_line--> [bs, 6*1024=6144] --view--> [bs, 1, 6, 1024] --unbind(2)--> 6 * [bs, 1, 1024]
        x = x + self.drop_path(
            self.attn(
                self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias,
                vis_attn_map=vis_attn_map).mul_(gamma1))
        
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        
        return x        # [16, 1, 1024]
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


# --------------------------------------------
# Softmax attention --> gated linear attention
# --------------------------------------------
class AdaLNGLA(nn.Module):
    def __init__(
        self, block_idx,
        # attn
        embed_dim: int, num_heads: int, norm_layer,
        # adaln
        cond_dim: int, shared_aln: bool,
        mlp_ratio=4., attn_l2_norm=False,                       # todo: VAR SA use attn_l2_norm on q, maybe add to GLA
        # drop
        drop=0., attn_drop=0., drop_path=0., last_drop_p=0,     # todo: VAR SA use attn_drop after output proj, maybe add to GLA
        fused_if_available=True,
        # GLA params
        gla_mode: str = 'fused_recurrent',
        gla_expand_k: float = 1.0,
        gla_expand_v: float = 1.0,
        gla_use_output_gate: bool = True,
        gla_hidden_act: str = 'swish',
        gla_elementwise_affine: bool = True,
        gla_norm_eps: float = 1e-6,
        gla_gate_logit_normalizer: int = 16,
        gla_gate_low_rank_dim: int = 16,
        gla_fuse_norm: bool = True,
    ):
        super(AdaLNGLA, self).__init__()

        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        # self.C, self.D = embed_dim, cond_dim
        # -->
        self.D_cond = cond_dim # Renamed to avoid conflict
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.attn = SelfAttention(
        #     block_idx=block_idx,
        #     embed_dim=embed_dim,
        #     num_heads=num_heads,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        #     attn_l2_norm=attn_l2_norm,
        #     flash_if_available=flash_if_available
        # )
        # -->
        self.attn = GatedLinearAttention(
            mode=gla_mode,
            hidden_size=embed_dim,
            expand_k=gla_expand_k,
            expand_v=gla_expand_v,
            num_heads=num_heads,
            num_kv_heads=None,      # Assuming MHA for VAR based on SelfAttention structure
            feature_map=None,
            use_short_conv=False,
            conv_size=4,
            conv_bias=False,
            use_output_gate=gla_use_output_gate,
            gate_fn=gla_hidden_act,
            elementwise_affine=gla_elementwise_affine,
            norm_eps=gla_norm_eps,
            gate_logit_normalizer=gla_gate_logit_normalizer,
            gate_low_rank_dim=gla_gate_low_rank_dim,
            clamp_min=None,
            fuse_norm=gla_fuse_norm,
            layer_idx=block_idx          # todo: how to use? # Pass block_idx as layer_idx for caching
        )

        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
            drop=drop,
            fused_if_available=fused_if_available
        )
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
            self,
            x,                      # train: [bs, 680, 1024], test: [16, 1, 1024], ..., [bs, 256, 1024]
            past_key_values,        # (Key, Value)
            cond_BD,                # [bs, 1024]
            attn_bias = None,        # train: [bs, 1, 680, 680], test: None

        ):   # C: embed_dim, D: cond_dim
        
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)     # cond_BD: [bs, 1024] --ada_line--> [bs, 6*1024=6144] --view--> [bs, 1, 6, 1024] --unbind(2)--> 6 * [bs, 1, 1024]
        
        # todo: support train
        # x = x + self.drop_path(
        #     self.attn(
        #         self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), 
        #         attn_bias=attn_bias).mul_(gamma1))
        # -->
        # todo: use SA KV Cache to get GLA recurrent state
        # past_key_values = torch.bmm(past_key_values[0].transpose(1, 2), past_key_values[1])
        if not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        attn_input = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        GatedLinearAttention.forward
        attn_output, _, _ = self.attn(
            hidden_states=attn_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        x = x + self.drop_path(attn_output.mul_(gamma1))
        
        x = x + self.drop_path(
            self.ffn(
                self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        
        return x        # [16, 1, 1024]
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(
            nn.SiLU(inplace=False), 
            nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
