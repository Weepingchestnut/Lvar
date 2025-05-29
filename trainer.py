from pprint import pformat
import time
from typing import List, Optional, Tuple, Union

from matplotlib.colors import ListedColormap
import numpy as np
import seaborn

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as tdist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

from models.ema import update_ema
from models.infinity.bitwise_self_correction import BitwiseSelfCorrection
from models.infinity.infinity_model import Infinity
from models.vae.quant import VectorQuantizer2
from models.vae.vqvae import VQVAE
from models.var.var_model import VAR

import utils.dist as dist
from utils import arg_util, misc, wandb_utils
from utils.amp_sc import AmpOptimizer
from utils.dynamic_resolution import dynamic_resolution_h_w
from utils.misc import MetricLogger, TensorboardLogger


fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


class VARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[torch.LongTensor] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: torch.Tensor = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def train_step(
        self,
        it: int,
        g_it: int,
        stepping: bool,
        metric_lg: MetricLogger,
        tb_lg: TensorboardLogger,
        inp_B3HW: torch.Tensor,
        label_B: Union[torch.LongTensor, torch.Tensor],
        prog_si: int, 
        prog_wp_it: float,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:    # 不走该if
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size      # batch_size, vocab_size=4096
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[torch.LongTensor] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)                     # [bs, 680]
        x_BLCv_wo_first_l: torch.Tensor = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)     # for teacher-forcing input
        
        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    if 0 <= prog_si < si: break
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
                tb_lg.update(head='AR_iter_schedule', prog_a_reso=self.resos[prog_si], prog_si=prog_si, prog_wp=prog_wp, step=g_it)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)


class InfinityTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, gpt_wo_ddp: Infinity, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool,
        dbg_unused=False, zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, other_args=None,
    ):
        super(InfinityTrainer, self).__init__()

        self.dbg_unused = dbg_unused    # 是否在训练步骤后检查并报告模型中未使用的参数（即梯度为 None 的参数）

        # VAE
        self.zero = zero                #  ZeRO (Zero Redundancy Optimizer), PyTorch FSDP (Fully Sharded Data Parallel)
        self.vae_type = vae_type

        # GPT / Infinity
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local, self.quantize_local = gpt, vae_local, vae_local.quantize
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema

        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.seq_len_each = []                  # 存储多尺度 VAE 输出的每个尺度的序列长度

        self.reweight_loss_by_scale = reweight_loss_by_scale        # 是否根据特征图的尺度来重新加权loss
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')

        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0

        self.is_visualizer = is_visualizer

        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled

        self.cached_state_not_ema = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio    # not use
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss              # 是否以及如何使用均衡损失（Equalization Loss）。如果非零，会根据特征图的尺度对损失进行加权，旨在平衡不同尺度 token 的重要性

        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device)
            cur = 0
            for raw_pn in raw_scale_schedule:
                l = raw_pn*raw_pn
                self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn) if self.eq_loss == 2 else raw_pn)
                cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        self.cmap_sim: ListedColormap = seaborn.color_palette('viridis', as_cmap=True)      # 一个 Matplotlib 的颜色映射表，用于可视化（例如生成相似度图），这里使用了 'viridis' 调色板

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

        self.generator = np.random.default_rng(0)
    
    @torch.no_grad()
    def eval_ep(self, ep: int, args: arg_util.Args, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0

        stt = time.time()       # record start time

        training = self.gpt_wo_ddp.training
        self.gpt_wo_ddp.eval()

        for inp, label_B in ld_val:
            B = label_B.shape[0]
            label_B = label_B.to(args.device, non_blocking=True)
            V = self.vae_local.vocab_size
            inp = inp.to(args.device, non_blocking=True)
            gt_ms_idx_Bl: List[torch.Tensor] = self.vae_local.get_GPT_ground_truth(inp)

            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)
            self.gpt_wo_ddp.forward
            logits_BLV = self.gpt_wo_ddp(label_B, self.quantize_local.fuse_multiscale_idx_as_gpt_inp_BL(gt_ms_idx_Bl))

            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.raw_last_l:].reshape(-1, V), gt_BL[:, -self.raw_last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.raw_last_l:].argmax(dim=-1) == gt_BL[:, -self.raw_last_l:]).sum() * (100/self.raw_last_l)
            tot += B
        
        self.gpt_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()

        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        inp_B3HW: torch.Tensor, text_cond_tuple: Union[torch.LongTensor, torch.Tensor], args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_B3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        T = 1 if inp_B3HW.dim() == 4 else inp_B3HW.shape[2]
        V = self.vae_local.vocab_size
        device = inp_B3HW.device

        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        # print("="*50)
        # print(type(scale_schedule))
        scale_schedule = [(min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
        # print(type(scale_schedule))

        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = scale_schedule
                    raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
            
            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_B3HW, raw_features, device)
            # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

            # truncate scales
            training_scales = args.always_training_scales
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum()
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]

            self.gpt_wo_ddp.forward  
            logits_BLV = self.gpt(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule[:training_scales]) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
            self.batch_size, self.seq_len = logits_BLV.shape[:2]

            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

            if self.reweight_loss_by_scale:
                lw = []
                # ------------------------------------------------------
                # print(scale_schedule[-1])     # tuple (1, 21, 14)
                # last_scale_area = np.sqrt(scale_schedule[-1].prod())
                # -->
                last_scale_area = np.sqrt(np.prod(scale_schedule[-1]))
                # ------------------------------------------------------
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
                lw = torch.tensor(lw, device=loss.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / self.seq_len
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # [backward]
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            B, seq_len = logits_BLV.shape[:2]
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1).mean(0)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1).mean(0)
            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float().mean(0)
            
            loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod()
                L_list.append(res_loss[start:end].mean().item())
                acc_bit_list.append(res_bit_acc[start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[start:end].mean().item() * 100.)
                ptr = end
            
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)
            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-4], metrics[-3], metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t)    # todo: Accm, Acct
            wandb_log_dict = {"Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t}
            for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                wandb_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                wandb_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                wandb_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
            wandb_utils.log(wandb_log_dict, step=g_it)
        
        return grad_norm_t, scale_log2_t

    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            if self.using_ema:  # TODO: fixme
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} missing:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} unexpected:  {unexpected}')
            
            if self.using_ema:
                if 'gpt_ema_for_vis' in state:
                    for pi, para in self.pi_para_copy_for_parallel_ema:
                        para.copy_(state['gpt_ema_for_vis'][self.gpt_opt.names[pi]])
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: load succeed')
                else:
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: key NOT FOUND in state!!')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
