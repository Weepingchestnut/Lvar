import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fastvar.utils import compute_merge
from models.hart.basic_hart import (FFN, FusedLlamaRotaryEmbedding1DWithPos,
                                    FusedLlamaRotaryEmbedding2DWithPos,
                                    LlamaMLP, LlamaRotaryEmbedding,
                                    LlamaRotaryEmbedding1D,
                                    MultiHeadCrossAttention, SelfAttention,
                                    context_pooling, get_position_ids,
                                    rotate_half)
from models.helpers import DropPath

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

# NOTE: the attn from torch is not the O(N^2) vanilla attn
def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
    if attn_mask is not None:
        attn.add_(attn_mask)
    return (
        F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
        if dropout_p > 0
        else attn.softmax(dim=-1)
    ) @ value


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# unsqueeze_dim=2 because by default our qk has shape [batch_size, seq_len, heads, head_dim]
def fastvar_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2, idx_fn=None):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor. [B, L, numHead, HeadDim]
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

    if idx_fn is not None and idx_fn.__name__ != 'do_nothing':
        rope_idx = idx_fn()
        print("rope_idx: ", rope_idx.shape)
        cos = torch.gather(cos, index=rope_idx.repeat(1,1,cos.shape[-1]), dim=1)
        sin = torch.gather(sin, index=rope_idx.repeat(1,1,sin.shape[-1]), dim=1)


    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim) # [2, len, 1, HeadDim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FastVARLlamaAttention(nn.Module):
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
        idx_fn=None,
        is_later_layer=None,):
        B, L, C = x.shape
        # [B, L, 2]
        if self.context_token == 0:
            position_ids = get_position_ids(
                B, self.patch_nums, x.device, si=si, m_maskgit=m_maskgit
            )
        else: # Here!!!
            # text to image
            # level 0 does not appear in the position_ids
            # since it is included in context tokens
            # should be 679 tokens for 16x16 latent w/ default 10-stage VAR
            if si == -1:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si, m_maskgit=m_maskgit)
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(-1).unsqueeze(-1)
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
        main_type = qkv.dtype
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

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        ################## Use naive rotary embedding ##################
        # apply position embedding to visual tokens
        if self.context_token == 0:
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

        q, k = fastvar_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze,idx_fn=idx_fn)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0


        if using_flash:
            # slow attn
            #q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            #oup = slow_attn(query=q,key=k,value=v,scale=self.scale,attn_mask=attn_bias,dropout_p=dropout_p,).transpose(1, 2).reshape(B, q.shape[2], C)

            # flash attn
            oup = flash_attn_func(
                q.to(dtype=main_type), # b,l,head,d
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, q.shape[1], C)
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
            ).view(B, q.shape[1], C)
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
                .reshape(B, q.shape[1], C)
            )

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"


class FastVARAdaLNSelfAttn(nn.Module):
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
                flash_if_available=flash_if_available,
            )
        else:
            self.attn = FastVARLlamaAttention(
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
                attn_l2_norm=attn_l2_norm,
            )
        if mlp_type == "gpt2":
            self.ffn = FFN(
                in_features=embed_dim,
                hidden_features=round(embed_dim * mlp_ratio),
                drop=drop,
                fused_if_available=fused_if_available,
                act_func=gpt2_mlp_act_func,
            )
        elif mlp_type == "llama":
            # MLP ratio = 4: mul 8 / 3
            self.ffn = LlamaMLP(
                in_features=embed_dim,
                hidden_features=int((embed_dim * mlp_ratio * 2) / 3 + 255) // 256 * 256,
                out_features=embed_dim,
                drop=drop,
                fused_if_available=fused_if_available,
            )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if not self.disable_aln:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        else:
            if self.shared_aln:
                self.scale_shift_table = nn.Parameter(
                    torch.randn(6, embed_dim) / embed_dim**0.5
                )
        self.fused_add_norm_fn = None
        self.use_cross_attn = use_cross_attn

        if self.use_cross_attn:
            self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)
        else:
            self.cross_attn = None

        self.previous_scale_cache_attn = None
        self.previous_scale_cache_ffn = None
        self.cache_scale_step = [36,36]

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
            context_mask=context_mask,
        )

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
        layer_idx =None,
    ):  # C: embed_dim, D: cond_dim
        # We achieve multi-token conditioning through LLM attention mask.
        if not self.disable_aln:
            condition = context_pooling(
                cond_BD, context_mask=context_mask, mode=self.sep_aln_pooling_mode
            ).unsqueeze(1)

            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_lin(condition).view(-1, 1, 6, self.C).unbind(2)
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
                    x[:, cond_BD.size(1) :], cond_BD, None
                )
            x = x + self.drop_path(
                self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
            )  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        else: # Here!!!
            is_later_layer = True
            merge_fn, unmerge_fn, idx_fn = compute_merge(x, is_later_layer=is_later_layer)
            if not self.shared_aln:
                attn_out = self.drop_path( # Here!!!
                    self.attn(self.ln_wo_grad(merge_fn(x)),
                                attn_bias=attn_bias,
                                context_position_ids=context_position_ids,
                                context_mask=context_mask,
                                si=si, is_later_layer=is_later_layer,
                                m_maskgit=m_maskgit,idx_fn=idx_fn))

                attn_out = unmerge_fn(attn_out,self.previous_scale_cache_attn)
                if x.shape[1] in [self.cache_scale_step[0]*self.cache_scale_step[1]]:
                    self.previous_scale_cache_attn = attn_out

                x = x + attn_out

                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )

                merge_fn, unmerge_fn, idx_fn = compute_merge(x, is_later_layer=is_later_layer)
                ffn_out = self.ffn(self.ln_wo_grad(merge_fn(x)))
                ffn_out = unmerge_fn(ffn_out,self.previous_scale_cache_ffn)
                if x.shape[1] in [self.cache_scale_step[0]*self.cache_scale_step[1]]:
                    self.previous_scale_cache_ffn = ffn_out
                x = x + self.drop_path(ffn_out) # Here!!!

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
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(
                    self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(
                        gamma2
                    )
                )
        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"
