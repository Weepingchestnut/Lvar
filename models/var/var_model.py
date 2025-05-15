import math
from typing import Optional, Tuple, Union
from functools import partial

import torch
import torch.nn as nn

import utils.dist as dist
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vae.vqvae import VQVAE, VectorQuantizer2
from .basic_var import AdaLNBeforeHead, AdaLNSelfAttn


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()

        # ------ 0. hyperparameters ------
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size                                # Cvae=32, V=4096
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads     # depth=16, C=1024, D=1024, num_heads=16

        self.cond_drop_rate = cond_drop_rate                                                    # cond_drop_rate=0.1
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums                                                # (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.L = sum(pn ** 2 for pn in self.patch_nums)                                         # L = 680
        self.first_l = self.patch_nums[0] ** 2                                                  # 1
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))                                          # [(0, 1), (1, 5), (5, 14), (14, 30), (30, 55), (55, 91), (91, 155), (155, 255), (255, 424), (424, 680)]
            cur += pn ** 2                                                                      # at last, cur=680
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1                                      # 9
        # self.rng = torch.Generator(device=dist.get_device())                                    # generate random numbers
        self.rng = torch.Generator(device='cuda')   # for c2i eval

        # ------ 1. input (word) embedding ------
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)                                          # 32 --> 1024

        # ------ 2. class embedding ------
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes                                                          # 1000
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())   # [1, 1000]
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)                             # [1001, 1024]
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))                     # [1, 1, 1024]
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # ------ 3. absolute position embedding ------
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)                                  # [[1, 1, 1024], [1, 4, 1024], ..., [1, 256, 1024]]
        pos_1LC = torch.cat(pos_1LC, dim=1)                     # 1, L, C [1, 680, 1024]
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)     # [10, 32]
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # ------ 4. backbone blocks ------
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate                                # 0.06666666666666667
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]     # [False, False, ..., False]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # ------ 5. attention mask used in training (for masking out the future) ------
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)   # [1, 680, 1]
        dT = d.transpose(1, 2)    # dT: 11L                                                                                 # [1, 1, 680]
        lvl_1L = dT[:, 0].contiguous()                                                                                      # [1, 680]
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)                          # [1, 1, 680, 680]
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())   # 

        # ------ 6. classifier head ------
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)                                                                               # [1024 --> 4096]
    
    def init_weights(self,
                     init_adaln=0.5,
                     init_adaln_gamma=1e-5,
                     init_head=0.02,
                     init_std=0.02,
                     conv_std_or_gain=0.02):
        
        if init_std < 0: 
            init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():    # m: VAR, Linear(in_features=32, out_features=1024, bias=True), Embedding(1001, 1024), Embedding(10, 1024), Identity(), ModuleList[], AdaLNSelfAttn, Identity(), SelfAttention, ...
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            # GLA and GLAMLP no such attribute
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:      # returns reconstructed image (B, 3, H, W) in [0, 1]
        """only used for inference, on autoregressive mode

        Args:
            B (int): batch size
            label_B (Optional[Union[int, torch.LongTensor]]): imagenet label; if None, randomly sampled
            g_seed (Optional[int], optional): random seed. Defaults to None.
            cfg (float, optional): classifier-free guidance ratio. Defaults to 4.
            top_k (int, optional): top-k sampling. Defaults to 900.
            top_p (float, optional): top-p sampling. Defaults to 0.95.
            more_smooth (bool, optional): smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking. Defaults to False.

        Returns:
            torch.Tensor: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: 
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))      # [2*batch, 1024]
        # cat: tensor([ 980,  980,  437,  437,   22,   22,  562,  562, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], device='cuda:0'), torch.Size([16])

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC        # [1, 680, 1024]
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]      # [1, 680, 1024]
        # sos: [16, 1024] -> [16, 1, 1024]; self.pos_start: [1, 1, 1024] -> [16, 1, 1024]; lvl_pos: [1, 680, 1024] -> [1, 1, 1024]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])       # [8, 32, 16, 16]

        for b in self.blocks:
            b.attn.kv_caching(True)

        for si, pn in enumerate(self.patch_nums):       # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)               # [16, 1024]
            x = next_token_map                                          # [16, 1, 1024]
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)                    # [16, 1, 4096]

            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]                                                    # [bs, seq_len(1->4->9...), Cvab(4096)]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]    # [bs, seq_len(1->4->9...)]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae                                       # [bs, seq_len(1->4->9...), Cvae(32)]
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)                                              # [bs, Cvae, h, w]: hw(1->2->3...)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks:
            b.attn.kv_caching(False)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """forward

        Args:
            label_B (torch.LongTensor): label_B
            x_BLCv_wo_first_l (torch.Tensor): teacher forcing input (B, self.L-self.first_l, self.Cvae)

        Returns:
            torch.Tensor: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)    # (0, 680)
        B = x_BLCv_wo_first_l.shape[0]

        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast('cuda', enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)    # 随机丢弃一些label，替换为特殊类标签1000
            sos = cond_BD = self.class_emb(label_B)                                                             # [bs, 1024]
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)     # [bs, 1, 1024]

            if self.prog_si == 0: 
                x_BLC = sos
            else: 
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]      # [1, 1, 680, 680]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)               # Identity: [bs, 1024]

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype                  # torch.float16
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)         # float32 --> float16
        attn_bias = attn_bias.to(dtype=main_type)                   # float32 --> float16

        AdaLNSelfAttn.forward
        # ------ Transformer blocks ------
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        # --------------------------------
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        
        return x_BLC    # logits BLV, V is vocab_size


if __name__ == '__main__':
    pass

