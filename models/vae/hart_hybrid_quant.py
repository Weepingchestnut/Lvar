import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import distributed as tdist
from torch.nn import functional as F

from models.vae.quant import Phi, PhiNonShared, PhiPartiallyShared, PhiShared


__all__ = [
    "VARQuantizer",
    "HARTHybridQuantizer",
]


class VARQuantizer(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self,
        vocab_size,
        Cvae,
        using_znorm,
        beta: float = 0.25,
        default_qresi_counts=0,
        v_patch_nums=None,
        quant_resi=0.5,
        share_quant_resi=4,  # share_quant_resi: args.qsr
        disable_quant_resi=False,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        # print(share_quant_resi, quant_resi)
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [
                    (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()
            )
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList(
                    [
                        (
                            Phi(Cvae, quant_resi)
                            if abs(quant_resi) > 1e-6
                            else nn.Identity()
                        )
                        for _ in range(share_quant_resi)
                    ]
                )
            )

        self.register_buffer(
            "ema_vocab_hit_SV",
            torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0),
        )
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1  # progressive training: not supported yet, prog_si always -1
        self.disable_quant_resi = disable_quant_resi

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(
                -abs(eini) / self.vocab_size, abs(eini) / self.vocab_size
            )

    def extra_repr(self) -> str:
        return f"{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}"

    # ===================== `forward` is only used in VAE training =====================
    def forward(
        self,
        f_BChw: torch.Tensor,
        patch_nums=None,
        ret_usages=False,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        if patch_nums is None:
            patch_nums = self.v_patch_nums

        idx_list = []

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(
                self.vocab_size, dtype=torch.float, device=f_BChw.device
            )
            SN = len(patch_nums)
            for si, pn in enumerate(patch_nums):  # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(
                        rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0),
                        dim=1,
                    )
                else:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    d_no_grad = torch.sum(
                        rest_NC.square(), dim=1, keepdim=True
                    ) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(
                        rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                    )  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    handler = tdist.all_reduce(hit_V, async_op=True)

                # calc loss
                idx_list.append(idx_N)
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = (
                    F.interpolate(
                        self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                        size=(H, W),
                        mode="bicubic",
                    ).contiguous()
                    if (si != SN - 1)
                    else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                )
                # if not (self.disable_quant_resi and si == SN - 1):
                #     h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest = f_rest - h_BChw

                if self.training:
                    handler.wait()
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss = (
                    mean_vq_loss
                    + F.mse_loss(f_hat.data, f_BChw).mul_(self.beta)
                    + F.mse_loss(f_hat, f_no_grad)
                )

            mean_vq_loss = mean_vq_loss * 1.0 / SN
            f_hat = (f_hat.data - f_no_grad) + (f_BChw)

        margin = 1
        if ret_usages:
            usages = (vocab_hit_V >= margin).float().mean().item() * 100
        else:
            usages = None
        return f_hat, usages, mean_vq_loss, idx_list

    # ===================== `forward` is only used in VAE training =====================

    def embed_to_fhat(
        self,
        ms_h_BChw: List[torch.Tensor],
        patch_nums=None,
        all_to_max_scale=True,
        last_one=False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if patch_nums is None:
            patch_nums = self.v_patch_nums

        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = patch_nums[-1]
        SN = len(patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training (where we'll interpolate every token map to the max scale), so it may cause some training-inference inconsistency
            # WARNING: this should only be used for experimental visualization
            f_hat = ms_h_BChw[0].new_zeros(
                B,
                self.Cvae,
                patch_nums[0],
                patch_nums[0],
                dtype=torch.float32,
            )
            for si, pn in enumerate(patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(
        self,
        f_BChw: torch.Tensor,
        to_fhat: bool,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        exception_stages: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[
        Union[torch.Tensor, torch.LongTensor]
    ]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)
        if exception_stages is not None:
            assert isinstance(exception_stages, dict)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            if 0 <= self.prog_si < si:
                break  # progressive training: not supported yet, prog_si always -1
            if exception_stages is None or (
                exception_stages is not None and si not in exception_stages
            ):
                # find the nearest embedding
                z_NC = (
                    F.interpolate(f_rest, size=(ph, pw), mode="area")
                    .permute(0, 2, 3, 1)
                    .reshape(-1, C)
                    if (si != SN - 1)
                    else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                )
                if self.using_znorm:
                    z_NC = F.normalize(z_NC, dim=-1)
                    idx_N = torch.argmax(
                        z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1
                    )
                else:
                    d_no_grad = torch.sum(
                        z_NC.square(), dim=1, keepdim=True
                    ) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(
                        z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                    )  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)
            else:
                assert exception_stages is not None and si in exception_stages
                assert len(exception_stages[si].shape) == 2
                assert exception_stages[si].shape[1] == ph * pw
                idx_N = exception_stages[si]

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = (
                F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(
                f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw)
            )

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(
        self, gt_ms_idx_Bl: List[torch.Tensor], patch_nums=None
    ) -> torch.Tensor:
        if patch_nums is None:
            patch_nums = self.v_patch_nums
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = patch_nums[-1]
        SN = len(patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break  # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(
                # last stage might be continuous
                (
                    self.embedding(gt_ms_idx_Bl[si])
                    if len(gt_ms_idx_Bl[si].shape) == 2
                    else gt_ms_idx_Bl[si]
                )
                .transpose_(1, 2)
                .view(B, C, pn_next, pn_next),
                size=(H, W),
                mode="bicubic",
            )
            f_hat.add_(self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = patch_nums[si + 1]
            next_scales.append(
                F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")
                .view(B, C, -1)
                .transpose(1, 2)
            )
        return (
            torch.cat(next_scales, dim=1) if len(next_scales) else None
        )  # cat BlCs to BLC, this should be float32

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(
        self,
        si: int,
        SN: int,
        f_hat: torch.Tensor,
        h_BChw: torch.Tensor,
        patch_nums=None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:  # only used in VAR inference
        if patch_nums is None:
            patch_nums = self.v_patch_nums
        HW = patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=(patch_nums[si + 1], patch_nums[si + 1]),
                mode="area",
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class HARTHybridQuantizer(VARQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        f_BChw: torch.Tensor,
        patch_nums=None,
        ret_usages=True,
        skip_continuous_prob=0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        if patch_nums is None:
            patch_nums = self.v_patch_nums

        idx_list = []

        # with torch.cuda.amp.autocast(enabled=False):
        with torch.amp.autocast("cuda", enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(
                self.vocab_size, dtype=torch.float, device=f_BChw.device
            )
            SN = len(patch_nums)
            for si, pn in enumerate(patch_nums):  # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(
                        rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0),
                        dim=1,
                    )
                else:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                    )
                    d_no_grad = torch.sum(
                        rest_NC.square(), dim=1, keepdim=True
                    ) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(
                        rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                    )  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    handler = tdist.all_reduce(hit_V, async_op=True)

                # calc loss
                idx_list.append(idx_N)
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(
                    self.beta
                ) + F.mse_loss(f_hat, f_no_grad)

            # optionally decode the continuous latent
            p = random.random()
            if p >= skip_continuous_prob:
                # skip the final stage with 50% probability
                h_BChw = f_rest.clone()
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        margin = 1
        if ret_usages:
            usages = (vocab_hit_V >= margin).float().mean().item() * 100
        else:
            usages = None
        return f_hat, usages, mean_vq_loss, idx_list

    def f_to_idxBl_and_frest(
        self,
        f_BChw: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        **kwargs,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        # return: [idx_Bl for si in [0, SN - 2], h_BChw for SN - 1]
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        idx_Bl_and_frest: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                self.embedding.weight.data.square(), dim=1, keepdim=False
            )
            d_no_grad.addmm_(
                z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
            )  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(
                self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                size=(H, W),
                mode="bicubic",
            ).contiguous()
            if si != SN - 1:
                # consistency
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            if si == SN - 1:
                f_rest_wo_last_discrete = f_rest.clone()
            f_rest.sub_(h_BChw)
            idx_Bl_and_frest.append(idx_N.reshape(B, ph * pw))

        h_BChw = f_rest
        f_hat = f_hat + h_BChw
        idx_Bl_and_frest.append(
            f_rest_wo_last_discrete.clone()
            .permute(0, 2, 3, 1)
            .reshape(-1, ph * pw, self.Cvae)
        )

        return idx_Bl_and_frest

    def get_next_autoregressive_input(
        self,
        si: int,
        SN: int,
        f_hat: torch.Tensor,
        h_BChw: torch.Tensor,
        patch_nums=None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:  # only used in VAR inference
        if patch_nums is None:
            patch_nums = self.v_patch_nums
        HW = patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=(patch_nums[si + 1], patch_nums[si + 1]),
                mode="area",
            )
        else:
            h = h_BChw
            f_hat.add_(h)
            return f_hat, f_hat
