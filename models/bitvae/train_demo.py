import argparse
import glob
import logging
import os
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from models.bitvae.d_vae import AutoEncoder, ImageDiscriminator, adopt_weight
from models.bitvae.data import ImageData


# ==============
# Training utils
# ==============
def get_last_ckpt(root_dir):
    if not os.path.exists(root_dir): return None
    ckpt_files = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ckpt'):
                num_iter = int(filename.split('.ckpt')[0].split('_')[-1])
                ckpt_files[num_iter]=os.path.join(dirpath, filename)
    iter_list = list(ckpt_files.keys())
    if len(iter_list) == 0: return None
    max_iter = max(iter_list)
    return ckpt_files[max_iter]


def is_torch_optimizer(obj):
    return isinstance(obj, optim.Optimizer)


def load_unstrictly(state_dict, model, loaded_keys=[]):
    missing_keys = []
    for name, param in model.named_parameters():
        if name in state_dict:
            try:
                param.data.copy_(state_dict[name])
            except:
                # print(f"{name} mismatch: param {name}, shape {param.data.shape}, state_dict shape {state_dict[name].shape}")
                missing_keys.append(name)
        elif name not in loaded_keys:
            missing_keys.append(name)
    return model, missing_keys


def resume_from_ckpt(state_dict, model_optims, load_optimizer=True):
    all_missing_keys = []
    # load weights first
    for k in model_optims:
        if model_optims[k] and (not is_torch_optimizer(model_optims[k])) and k in state_dict:
            model_optims[k], missing_keys = load_unstrictly(state_dict[k], model_optims[k])
            all_missing_keys += missing_keys
        
    if len(all_missing_keys) == 0 and load_optimizer:
        print("Loading optimizer states")
        for k in model_optims: 
            if model_optims[k] and is_torch_optimizer(model_optims[k]) and k in state_dict:
                model_optims[k].load_state_dict(state_dict[k])
    else:
        print(f"missing weights: {all_missing_keys}, do not load optimzer states")
    return model_optims, state_dict["step"]


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def get_disc_loss(disc_loss_type):
    if disc_loss_type == 'vanilla':
        disc_loss = vanilla_d_loss
    elif disc_loss_type == 'hinge':
        disc_loss = hinge_d_loss
    return disc_loss


def lecam_reg_zero(real_pred, fake_pred, thres=0.1):
    # avoid logits get too high
    assert real_pred.ndim == 0
    reg = torch.mean(F.relu(torch.abs(real_pred) - thres).pow(2)) + \
    torch.mean(F.relu(torch.abs(fake_pred) - thres).pow(2))
    return reg


def reduce_losses(loss_dict, dst=0):
    loss_names = list(loss_dict.keys())
    loss_tensor = torch.stack([loss_dict[name] for name in loss_names])

    dist.reduce(loss_tensor, dst=dst, op=dist.ReduceOp.SUM)
    # Only average the loss values on the destination rank
    if dist.get_rank() == dst:
        loss_tensor /= dist.get_world_size()
        averaged_losses = {name: loss_tensor[i].item() for i, name in enumerate(loss_names)}
    else:
        averaged_losses = {name: None for name in loss_names}
    
    return averaged_losses


def rank_zero_only(fn):
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)
    return wrapped_fn


@rank_zero_only
def average_losses(loss_dict_list):
    sum_dict = {}
    count_dict = {}
    for loss_dict in loss_dict_list:
        for key, value in loss_dict.items():
            if key in sum_dict:
                sum_dict[key] += value
                count_dict[key] += 1
            else:
                sum_dict[key] = value
                count_dict[key] = 1

    avg_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}
    return avg_dict


# =============
# distributed
# =============
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# =======
# logger
# =======
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        existing_logs = glob.glob(os.path.join(logging_dir, 'log_*.txt'))
        log_numbers = [int(log.split('.txt')[0].split('_')[-1]) for log in existing_logs]
        next_log_number = max(log_numbers) + 1 if log_numbers else 1
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log_{next_log_number}.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# ==========
# arguments
# ==========
def add_model_specific_args(args, parser):
    if args.tokenizer == "flux":
        parser = AutoEncoder.add_model_specific_args(parser) # flux config
        d_vae_model = AutoEncoder
    else:
        raise NotImplementedError
    return args, parser, d_vae_model


class MainArgs:
    @staticmethod
    def add_main_args(parser):
        # training
        parser.add_argument('--max_steps', type=int, default=1e6)
        parser.add_argument('--log_every', type=int, default=1)
        parser.add_argument('--visu_every', type=int, default=1000)
        parser.add_argument('--ckpt_every', type=int, default=1000)
        parser.add_argument('--default_root_dir', type=str, required=True)
        parser.add_argument('--multiscale_training', action="store_true")

        # optimization
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.95)
        parser.add_argument('--warmup_steps', type=int, default=0)
        parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "AdamW"])
        parser.add_argument('--disc_optim_type', type=str, default=None, choices=[None, "rmsprop"])
        parser.add_argument('--lr_min', type=float, default=0.)
        parser.add_argument('--warmup_lr_init', type=float, default=0.)
        parser.add_argument('--max_grad_norm', type=float, default=1.0)
        parser.add_argument('--max_grad_norm_disc', type=float, default=1.0)
        parser.add_argument('--disable_sch', action="store_true")

        # basic d_vae config
        parser.add_argument('--patch_size', type=int, default=8)
        parser.add_argument('--codebook_dim', type=int, default=16)
        parser.add_argument('--quantizer_type', type=str, default='MultiScaleLFQ')

        parser.add_argument('--new_quant', action="store_true") # use new quantization (fix the potential bugs of the old quantizer)
        parser.add_argument('--use_decay_factor', action="store_true")
        parser.add_argument('--use_stochastic_depth', action="store_true")
        parser.add_argument("--drop_rate", type=float, default=0.0)
        parser.add_argument('--schedule_mode', type=str, default="original", choices=["original", "dynamic", "dense", "same1", "same2", "same3", "half", "dense_f8"])
        parser.add_argument('--lr_drop', nargs='*', type=int, default=None, help="A list of numeric values. Example: --values 270 300")
        parser.add_argument('--lr_drop_rate', type=float, default=0.1)
        parser.add_argument('--keep_first_quant', action="store_true")
        parser.add_argument('--keep_last_quant', action="store_true")
        parser.add_argument('--remove_residual_detach', action="store_true")
        parser.add_argument('--use_out_phi', action="store_true")
        parser.add_argument('--use_out_phi_res', action="store_true")
        parser.add_argument('--lecam_weight', type=float, default=0.05)
        parser.add_argument('--perceptual_model', type=str, default="vgg16", choices=["vgg16"])
        parser.add_argument('--base_ch_disc', type=int, default=64)
        parser.add_argument('--random_flip', action="store_true")
        parser.add_argument('--flip_prob', type=float, default=0.5)
        parser.add_argument('--flip_mode', type=str, default="stochastic", choices=["stochastic"])
        parser.add_argument('--max_flip_lvl', type=int, default=1)
        parser.add_argument('--not_load_optimizer', action="store_true")
        parser.add_argument('--use_lecam_reg_zero', action="store_true")
        parser.add_argument('--rm_downsample', action="store_true")
        parser.add_argument('--random_flip_1lvl', action="store_true")
        parser.add_argument('--flip_lvl_idx', type=int, default=0)
        parser.add_argument('--drop_when_test', action="store_true")
        parser.add_argument('--drop_lvl_idx', type=int, default=None)
        parser.add_argument('--drop_lvl_num', type=int, default=0)
        parser.add_argument('--compute_all_commitment', action="store_true")
        parser.add_argument('--disable_codebook_usage', action="store_true")
        parser.add_argument('--random_short_schedule', action="store_true")
        parser.add_argument('--short_schedule_prob', type=float, default=0.5)
        parser.add_argument('--disable_flip_prob', type=float, default=0.0)
        parser.add_argument('--zeta', type=float, default=1.0) # entropy penalty weight
        parser.add_argument('--disable_codebook_usage_bit', action="store_true")
        parser.add_argument('--gamma', type=float, default=1.0) # loss weight of H(E[p(c|u)])
        parser.add_argument('--uniform_short_schedule', action="store_true")

        # discriminator config
        parser.add_argument('--dis_warmup_steps', type=int, default=0)
        parser.add_argument('--dis_lr_multiplier', type=float, default=1.)
        parser.add_argument('--dis_minlr_multiplier', action="store_true")
        parser.add_argument('--disc_layers', type=int, default=3)
        parser.add_argument('--discriminator_iter_start', type=int, default=0)
        parser.add_argument('--disc_pretrain_iter', type=int, default=0)
        parser.add_argument('--disc_optim_steps', type=int, default=1)
        parser.add_argument('--disc_warmup', type=int, default=0)
        parser.add_argument('--disc_pool', type=str, default="no", choices=["no", "yes"])
        parser.add_argument('--disc_pool_size', type=int, default=1000)

        # loss
        parser.add_argument("--recon_loss_type", type=str, default='l1', choices=['l1', 'l2'])
        parser.add_argument('--image_gan_weight', type=float, default=1.0)
        parser.add_argument('--image_disc_weight', type=float, default=0.)
        parser.add_argument('--l1_weight', type=float, default=4.0)
        parser.add_argument('--gan_feat_weight', type=float, default=0.0)
        parser.add_argument('--perceptual_weight', type=float, default=0.0)
        parser.add_argument('--kl_weight', type=float, default=0.)
        parser.add_argument('--lfq_weight', type=float, default=0.)
        parser.add_argument('--entropy_loss_weight', type=float, default=0.1)
        parser.add_argument('--commitment_loss_weight', type=float, default=0.25)
        parser.add_argument('--diversity_gamma', type=float, default=1)
        parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group', "no"])
        parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])

        # acceleration
        parser.add_argument('--use_checkpoint', action="store_true")
        parser.add_argument('--precision', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--encoder_dtype', type=str, default="fp32", choices=['fp32', 'bf16']) # disable fp16
        parser.add_argument('--upcast_tf32', action="store_true")

        # initialization
        parser.add_argument('--tokenizer', type=str, default='flux', choices=["flux"])
        parser.add_argument('--pretrained', type=str, default=None)
        parser.add_argument('--pretrained_mode', type=str, default="full", choices=['full'])

        # misc
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--bucket_cap_mb', type=int, default=40) # DDP
        parser.add_argument('--manual_gc_interval', type=int, default=1000) # DDP

        return parser


def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    # 添加项目通用的参数 (如 batch_size, lr 等)
    parser = MainArgs.add_main_args(parser)
    # 添加数据相关的参数 (如 data_path)
    parser = ImageData.add_data_specific_args(parser)
    # 解析已知参数，用于后续动态添加模型参数
    args, unknown = parser.parse_known_args()
    # 根据 args 中的 model_type 添加特定模型的参数
    args, parser, d_vae_model = add_model_specific_args(args, parser)
    args = parser.parse_args()
    
    args.resolution = (args.resolution[0], args.resolution[0]) if len(args.resolution) == 1 else args.resolution # init resolution
    
    print(f"{args.default_root_dir=}")
    
    # Setup DDP:
    init_distributed_mode(args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.default_root_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders
        checkpoint_dir = f"{args.default_root_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.default_root_dir)
        logger.info(f"Experiment directory created at {args.default_root_dir}")

        import wandb
        wandb_project = "VQVAE"
        wandb.init(
            project=wandb_project,
            name=os.path.basename(os.path.normpath(args.default_root_dir)),
            dir=args.default_root_dir,
            config=args,
            mode="offline" if args.debug else "online"
        )
    else:
        logger = create_logger(None)
    
    # init dataloader
    data = ImageData(args)
    dataloaders = data.train_dataloader()
    dataloader_iters = [iter(loader) for loader in dataloaders]
    data_epochs = [0 for _ in dataloaders]
    
    # init model    
    d_vae = d_vae_model(args).to(device)
    d_vae.logger = logger
    image_disc = ImageDiscriminator(args).to(device)
    
    # init optimizers and schedulers
    if args.optim_type == "Adam":
        optim = torch.optim.Adam
    elif args.optim_type == "AdamW":
        optim = torch.optim.AdamW
    if args.disc_optim_type is None:
        disc_optim = optim
    elif args.disc_optim_type == "rmsprop":
        disc_optim = torch.optim.RMSprop
    opt_vae = optim(d_vae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    if disc_optim == torch.optim.RMSprop:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=args.lr * args.dis_lr_multiplier)
    else:
        opt_image_disc = disc_optim(image_disc.parameters(), lr=args.lr * args.dis_lr_multiplier, betas=(args.beta1, args.beta2))
    
    if args.disable_sch:
        # scheduler_list = [None, None]
        sch_vae, sch_image_disc = None, None
    
    model_optims = {
        "vae" : d_vae,
        "image_disc" : image_disc,
        "opt_vae" : opt_vae,
        "opt_image_disc" : opt_image_disc,
        "sch_vae" : sch_vae,
        "sch_image_disc" : sch_image_disc,
    }
    
    # resume from default_root_dir
    ckpt_path = None
    assert not args.default_root_dir is None # required argument
    ckpt_path = get_last_ckpt(args.default_root_dir)
    init_step = 0
    load_optimizer = not args.not_load_optimizer
    if ckpt_path:
        logger.info(f"Resuming from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model_optims, init_step = resume_from_ckpt(state_dict, model_optims, load_optimizer=True)
    # load pretrained weights
    elif args.pretrained is not None:
        state_dict = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        if args.pretrained_mode == "full":
            model_optims, _ = resume_from_ckpt(state_dict, model_optims, load_optimizer=load_optimizer)
        logger.info(f"Successfully loaded ckpt {args.pretrained}, pretrained_mode {args.pretrained_mode}")
    
    # DDP 封装
    d_vae = DDP(d_vae.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    image_disc = DDP(image_disc.to(device), device_ids=[args.gpu], bucket_cap_mb=args.bucket_cap_mb)
    # 获取 GAN 损失函数 (通常是 Hinge Loss)
    disc_loss = get_disc_loss(args.disc_loss_type) # hinge loss by default
    
    # mutil-scale training setting, for stage2
    if args.multiscale_training:
        scale_idx_list = np.load('models/bitvae/random_numbers.npy') # load pre-computed scale_idx in each iteration
    
    start_time = time.time()
    for global_step in range(init_step, args.max_steps):
        loss_dicts = []
        
        # --- 日志控制：判别器预热等 ---
        if global_step == args.discriminator_iter_start - args.disc_pretrain_iter:
            logging.info(f"discriminator begins pretraining ")
        if global_step == args.discriminator_iter_start:
            log_str = "add GAN loss into training"
            if args.disc_pretrain_iter > 0:
                log_str += ", discriminator ends pretraining"
            logging.info(log_str)
        
        # get data
        for idx in range(len(dataloader_iters)):
            try:
                _batch = next(dataloader_iters[idx])
            except StopIteration:
                # Epoch 结束，重置 dataloader
                data_epochs[idx] += 1
                logger.info(f"Reset the {idx}th dataloader as epoch {data_epochs[idx]}")
                dataloaders[idx].sampler.set_epoch(data_epochs[idx])
                dataloader_iters[idx] = iter(dataloaders[idx]) # update dataloader iter
                _batch = next(dataloader_iters[idx])
            except Exception as e:
                raise e
            
            x = _batch["image"]         # [batch, 3, 256, 256]
            _type = _batch["type"][0]   # 'image'
            
            if args.multiscale_training:
                # data processing for multi-scale training
                scale_idx = scale_idx_list[global_step]     # 根据 step 从预设列表中获取 scale_idx
                # 动态调整 x 的分辨率 (interpolate) 和 batch size (切片)
                if scale_idx == 0:
                    # 256x256 batch=8
                    x = F.interpolate(x, size=(256, 256), mode='area')
                elif scale_idx == 1:
                    # 512x512 batch=4
                    rdn_idx = torch.randperm(len(x))[:4] # without replacement
                    x = x[rdn_idx]
                    x = F.interpolate(x, size=(512, 512), mode='area')
                elif scale_idx == 2:
                    # 1024x1024 batch=2
                    rdn_idx = torch.randperm(len(x))[:2] # without replacement
                    x = x[rdn_idx]
                else:
                    raise ValueError(f"scale_idx {scale_idx} is not supported")
            
            # --- training VAE ---
            if _type == "image":
                # VAE 前向传播：计算重构图和 VAE 内部损失 (L1/L2, Perceptual, Commitment, GAN_G)
                x_recon, flat_frames, flat_frames_recon, vae_loss_dict = d_vae(x, global_step, image_disc=image_disc)
            g_loss = sum(vae_loss_dict.values())
            opt_vae.zero_grad()
            g_loss.backward()
            
            # 梯度裁剪与参数更新
            if not ((global_step+1) % args.ckpt_every) == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(d_vae.parameters(), args.max_grad_norm)
                if not sch_vae is None:
                    sch_vae.step(global_step)
                elif args.lr_drop and global_step in args.lr_drop:
                    logger.info(f"multiply lr of VQ-VAE by {args.lr_drop_rate} at iteration {global_step}")
                    for opt_vae_param_group in opt_vae.param_groups:
                        opt_vae_param_group["lr"] = opt_vae_param_group["lr"] * args.lr_drop_rate
                opt_vae.step()
            opt_vae.zero_grad() # free memory
            
            # --- training Discriminator ---
            disc_loss_dict = {}
            # disc_factor = 0 before (args.discriminator_iter_start - args.disc_pretrain_iter) 判别器权重控制 (预热阶段权重为 0)
            disc_factor = adopt_weight(global_step, threshold=args.discriminator_iter_start - args.disc_pretrain_iter)
            discloss = d_image_loss = torch.tensor(0.).to(x.device)
            ### enable pool warmup
            # 判别器循环 (通常 step 数为 1)
            for disc_step in range(args.disc_optim_steps): # train discriminator
                require_optim = False
                if _type == "image" and args.image_disc_weight > 0: # train image discriminator
                    require_optim = True
                    logits_image_real = image_disc(x, pool_name="real")                     # 判别真图
                    logits_image_fake = image_disc(x_recon.detach(), pool_name="fake")      # 判别假图 (使用 detach 过的 x_recon，不传梯度给 VAE)
                    d_image_loss = disc_loss(logits_image_real, logits_image_fake)          # 计算 Hinge Loss
                    disc_loss_dict["train/logits_image_real"] = logits_image_real.mean().detach()
                    disc_loss_dict["train/logits_image_fake"] = logits_image_fake.mean().detach()
                    disc_loss_dict["train/d_image_loss"] = d_image_loss.mean().detach()
                    discloss = d_image_loss * args.image_disc_weight
                    opt_discs, sch_discs = [opt_image_disc], [sch_image_disc]
                    # LeCam 正则化 (稳定训练)
                    if global_step >= args.discriminator_iter_start and args.use_lecam_reg_zero:
                        lecam_zero_loss = lecam_reg_zero(logits_image_real.mean(), logits_image_fake.mean())
                        disc_loss_dict["train/lecam_zero_loss"] = lecam_zero_loss.mean().detach()
                        discloss += lecam_zero_loss * args.lecam_weight
                discloss = disc_factor * discloss       # 应用预热权重
                
                # 判别器反向传播与更新
                if require_optim:
                    for opt_disc in opt_discs:
                        opt_disc.zero_grad()
                    discloss.backward()

                    if not ((global_step+1) % args.ckpt_every) == 0:
                        if args.max_grad_norm_disc > 0: # by default, 1.0
                            torch.nn.utils.clip_grad_norm_(image_disc.parameters(), args.max_grad_norm_disc)
                        for sch_disc in sch_discs:
                            if not sch_disc is None:
                                sch_disc.step(global_step)
                            elif args.lr_drop and global_step in args.lr_drop:
                                for opt_disc in opt_discs:
                                    logger.info(f"multiply lr of discriminator by {args.lr_drop_rate} at iteration {global_step}")
                                    for opt_disc_param_group in opt_disc.param_groups:
                                        opt_disc_param_group["lr"] = opt_disc_param_group["lr"] * args.lr_drop_rate
                        for opt_disc in opt_discs:
                            opt_disc.step()
                    for opt_disc in opt_discs:
                        opt_disc.zero_grad() # free memory
            
            loss_dict = {**vae_loss_dict, **disc_loss_dict}
            if (global_step+1) % args.log_every == 0:
                reduced_loss_dict = reduce_losses(loss_dict)
            else:
                reduced_loss_dict = {}
            loss_dicts.append(reduced_loss_dict)
        
        if (global_step+1) % args.log_every == 0:
            avg_loss_dict = average_losses(loss_dicts)
            torch.cuda.synchronize()
            end_time = time.time()
            iter_speed = (end_time - start_time) / args.log_every
            if rank == 0:
                for key, value in avg_loss_dict.items():
                    wandb.log({key: value}, step=global_step)
                # writing logs
                logger.info(f'global_step={global_step}, precepetual_loss={avg_loss_dict.get("train/perceptual_loss",0):.4f}, recon_loss={avg_loss_dict.get("train/recon_loss",0):.4f}, commitment_loss={avg_loss_dict.get("train/commitment_loss",0):.4f}, logit_r={avg_loss_dict.get("train/logits_image_real",0):.4f}, logit_f={avg_loss_dict.get("train/logits_image_fake",0):.4f}, L_disc={avg_loss_dict.get("train/d_image_loss",0):.4f}, iter_speed={iter_speed:.2f}s')
            start_time = time.time()
        
        if (global_step+1) % args.ckpt_every == 0 and global_step != init_step:
            if rank == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_step_{global_step}.ckpt')
                save_dict = {}
                for k in model_optims:
                    save_dict[k] = None if model_optims[k] is None \
                        else model_optims[k].module.state_dict() if hasattr(model_optims[k], "module") \
                        else model_optims[k].state_dict()
                torch.save({
                    'step': global_step,
                    **save_dict,
                }, checkpoint_path)
                logger.info(f'Checkpoint saved at step {global_step}')    


if __name__ == "__main__":
    main()