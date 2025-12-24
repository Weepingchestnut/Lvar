import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as checkpoint

from models.bitvae.lpips import LPIPS
from models.bitvae.ms_bsq import MultiScaleBSQ


ptdtype = {None: torch.float32, 'fp32': torch.float32, 'bf16': torch.bfloat16}


def swish(x: Tensor) -> Tensor:
    try:
        return x * torch.sigmoid(x)
    except:
        device = x.device
        x = x.cpu().pin_memory()
        return (x*torch.sigmoid(x)).to(device=device)


def adopt_weight(global_step, threshold=0, value=0., warmup=0):
    if global_step < threshold or threshold < 0:
        weight = value
    else:
        weight = 1
        if global_step - threshold < warmup:
            weight = min((global_step - threshold) / warmup, 1)
    return weight


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.stride = stride
        self.kernel_size = kernel_size
        
    def forward(self, x):
        return self.conv(x)


class Normalize(nn.Module):
    def __init__(self, in_channels, norm_type):
        super().__init__()
        assert norm_type in ['group', 'batch', "no"]
        if norm_type == 'group':
            if in_channels % 32 == 0:
                self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
            elif in_channels % 24 == 0: 
                self.norm = nn.GroupNorm(num_groups=24, num_channels=in_channels, eps=1e-6, affine=True)
            else:
                raise NotImplementedError
        elif norm_type == 'batch':
            self.norm = nn.SyncBatchNorm(in_channels, track_running_stats=False) # Runtime Error: grad inplace if set track_running_stats to True
        elif norm_type == 'no':
            self.norm = nn.Identity()
    
    def forward(self, x):
        assert x.ndim == 4
        x = self.norm(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type)

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels, norm_type)

        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, spatial_down=False):
        super().__init__()
        assert spatial_down == True
        self.pad = (0, 1, 0, 1)
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        x = nn.functional.pad(x, self.pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, spatial_up=False):
        super().__init__()
        assert spatial_up == True

        self.scale_factor = 2
        self.conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        in_channels = 3,
        patch_size=8,
        norm_type='group',
        use_checkpoint=False,
    ):
        super().__init__()
        self.max_down = np.log2(patch_size)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint
        # downsampling
        # self.conv_in = Conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        self.conv_in = Conv(in_channels, ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn

            spatial_down = True if i_level < self.max_down else False
            if spatial_down:
                down.downsample = Downsample(block_in, spatial_down=spatial_down)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = Conv(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, return_hidden=False):
        if not self.use_checkpoint:
            return self._forward(x, return_hidden=return_hidden)
        else:
            return checkpoint.checkpoint(self._forward, x, return_hidden, use_reentrant=False)

    def _forward(self, x: Tensor, return_hidden=False) -> Tensor:
        # downsampling
        h0 = self.conv_in(x)
        hs = [h0]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], "downsample"):
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        hs_mid = [h]
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)
        hs_mid.append(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        if return_hidden:
            return h, hs, hs_mid
        else:
            return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        out_ch = 3, 
        patch_size=8,
        norm_type="group",
        use_checkpoint=False,
    ):
        super().__init__()
        self.max_up = np.log2(patch_size)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.use_checkpoint = use_checkpoint

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = Conv(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, norm_type=norm_type)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, norm_type=norm_type))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            # https://github.com/black-forest-labs/flux/blob/b4f689aaccd40de93429865793e84a734f4a6254/src/flux/modules/autoencoder.py#L228
            spatial_up = True if 1 <= i_level <= self.max_up else False
            if spatial_up:
                up.upsample = Upsample(block_in, spatial_up=spatial_up)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = Conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        if not self.use_checkpoint:
            return self._forward(z)
        else:
            return checkpoint.checkpoint(self._forward, z, use_reentrant=False)

    def _forward(self, z: Tensor) -> Tensor:
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(
            ch=args.base_ch,
            ch_mult=args.encoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            use_checkpoint=args.use_checkpoint,
        )
        self.decoder = Decoder(
            ch=args.base_ch,
            ch_mult=args.decoder_ch_mult,
            num_res_blocks=args.num_res_blocks,
            z_channels=args.codebook_dim,
            patch_size=args.patch_size,
            use_checkpoint=args.use_checkpoint,
        )

        self.gan_feat_weight = args.gan_feat_weight
        self.recon_loss_type = args.recon_loss_type
        self.l1_weight = args.l1_weight
        self.kl_weight = args.kl_weight
        self.lfq_weight = args.lfq_weight
        self.image_gan_weight = args.image_gan_weight # image GAN loss weight
        self.perceptual_weight = args.perceptual_weight

        self.compute_all_commitment = args.compute_all_commitment # compute commitment between input and rq-output

        self.perceptual_model = LPIPS(upcast_tf32=args.upcast_tf32).eval()

        if args.quantizer_type == 'MultiScaleBSQ':
            self.quantizer = MultiScaleBSQ(
                dim = args.codebook_dim,                        # this is the input feature dimension, defaults to log2(codebook_size) if not defined
                entropy_loss_weight = args.entropy_loss_weight, # how much weight to place on entropy loss
                diversity_gamma = args.diversity_gamma,         # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
                commitment_loss_weight=args.commitment_loss_weight, # loss weight of commitment loss
                new_quant=args.new_quant,
                use_decay_factor=args.use_decay_factor,
                use_stochastic_depth=args.use_stochastic_depth,
                drop_rate=args.drop_rate,
                schedule_mode=args.schedule_mode,
                keep_first_quant=args.keep_first_quant,
                keep_last_quant=args.keep_last_quant,
                remove_residual_detach=args.remove_residual_detach,
                use_out_phi=args.use_out_phi,
                use_out_phi_res=args.use_out_phi_res,
                random_flip = args.random_flip,
                flip_prob = args.flip_prob,
                flip_mode = args.flip_mode,
                max_flip_lvl = args.max_flip_lvl,
                random_flip_1lvl = args.random_flip_1lvl,
                flip_lvl_idx = args.flip_lvl_idx,
                drop_when_test = args.drop_when_test,
                drop_lvl_idx = args.drop_lvl_idx,
                drop_lvl_num = args.drop_lvl_num,
                random_short_schedule = args.random_short_schedule,
                short_schedule_prob = args.short_schedule_prob,
                disable_flip_prob = args.disable_flip_prob,
                zeta = args.zeta,
                gamma = args.gamma,
                uniform_short_schedule = args.uniform_short_schedule
            )
        else:
            raise NotImplementedError(f"{args.quantizer_type} not supported")
        self.commitment_loss_weight = args.commitment_loss_weight

    def forward(self, x, global_step, image_disc=None, is_train=True):
        assert x.ndim == 4 # assert input data is image

        enc_dtype = ptdtype[self.args.encoder_dtype]    # torch.float32

        with torch.amp.autocast("cuda", dtype=enc_dtype):
            h = self.encoder(x, return_hidden=False) # B C H W  [batch, 3, 256, 256] --> [batch, 16, 16, 16]
        h = h.to(dtype=torch.float32)

        # Multiscale LFQ
        z, all_indices, all_bit_indices, all_loss = self.quantizer(h)
        # print(torch.unique(torch.round(z * 10**4)/10**4)) # keep 4 decimal places
        x_recon = self.decoder(z)
        vq_output = {
            "commitment_loss": torch.mean(all_loss) * self.lfq_weight, # here commitment loss is sum of commitment loss and entropy penalty
            "encodings": all_indices,
            "bit_encodings": all_bit_indices,
        }
        if self.compute_all_commitment:
            # compute commitment loss between input and rq-output
            vq_output["all_commitment_loss"] = F.mse_loss(h, z.detach(), reduction="mean") * self.commitment_loss_weight * self.lfq_weight
        else:
            # disable backward prop
            vq_output["all_commitment_loss"] = F.mse_loss(h.detach(), z.detach(), reduction="mean") * self.commitment_loss_weight * self.lfq_weight

        assert x.shape == x_recon.shape, f"x.shape {x.shape}, x_recon.shape {x_recon.shape}"

        if is_train == False:
            return x_recon, vq_output

        if self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        else:
            recon_loss = F.mse_loss(x_recon, x) * self.l1_weight

        flat_frames = x
        flat_frames_recon = x_recon

        perceptual_loss = self.perceptual_model(flat_frames, flat_frames_recon).mean() * self.perceptual_weight

        loss_dict = {
            "train/perceptual_loss": perceptual_loss,
            "train/recon_loss": recon_loss,
            "train/commitment_loss": vq_output['commitment_loss'],
            "train/all_commitment_loss": vq_output['all_commitment_loss'],
        }

        ### GAN loss
        disc_factor = adopt_weight(global_step, threshold=self.args.discriminator_iter_start, warmup=self.args.disc_warmup)
        if self.image_gan_weight > 0: # image GAN loss
            logits_image_fake = image_disc(flat_frames_recon)
            g_image_loss = -torch.mean(logits_image_fake) * self.image_gan_weight * disc_factor # disc_factor=0 before self.args.discriminator_iter_start
            loss_dict["train/g_image_loss"] = g_image_loss

        return (x_recon.detach(), flat_frames.detach(), flat_frames_recon.detach(), loss_dict)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--codebook_size", type=int, default=16384)

        parser.add_argument("--base_ch", type=int, default=128)
        parser.add_argument("--num_res_blocks", type=int, default=2)  # num_res_blocks for encoder, num_res_blocks+1 for decoder
        parser.add_argument("--encoder_ch_mult", type=int, nargs='+', default=[1, 1, 2, 2, 4])
        parser.add_argument("--decoder_ch_mult", type=int, nargs='+', default=[1, 1, 2, 2, 4])
        return parser


class DiscriminatorPool:
    def __init__(self, pool_size):
        self.pool_size = int(pool_size)
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            if self.num_imgs < self.pool_size:
                self.images.append(image)
                self.num_imgs += 1
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.pool_size - 1)
                    tmp = self.images[i].clone()
                    self.images[i] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.stack(return_images)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_type = "batch"
        use_bias = norm_type != "batch"

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                Normalize(ndf * nf_mult, norm_type=norm_type),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            Normalize(ndf * nf_mult, norm_type=norm_type),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):    
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class ImageDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.discriminator = NLayerDiscriminator(ndf=args.base_ch_disc) # by default using PatchGAN
        self.disc_pool = args.disc_pool # be default "no"
        if args.disc_pool == "yes":
            self.real_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
            self.fake_pool = DiscriminatorPool(pool_size=args.batch_size[0] * args.disc_pool_size)
    
    def forward(self, x, pool_name=None):
        if pool_name and self.disc_pool == "yes":
            assert pool_name in ["real", "fake"]
            if pool_name == "real":
                x = self.real_pool.query(x)
            elif pool_name == "fake":
                x = self.fake_pool.query(x)
        # by default without pool
        return self.discriminator(x)
