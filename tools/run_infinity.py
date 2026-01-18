import argparse
import hashlib
import cv2
import os

import imageio
from timm import create_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp

import numpy as np
import PIL.Image as PImage
from PIL import Image, ImageEnhance
import re
import shutil
import time
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import autocast
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer, T5Config, T5EncoderModel, T5TokenizerFast

from models.infinity.infinity_model import Infinity
from models.fastvar.fastvar_model import FastVAR_Infinity
from models.skipvar.skipvar_model import SkipVAR_Infinity
from models.sparsevar.sparsevar_model import SparseVAR_Infinity
from models.scalekv.scale_kv import enable_scale_kv
from models.sparvar.sparvar_model import SparVAR_Infinity
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def extract_key_val(text):
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val


def encode_prompt(text_tokenizer, text_encoder, prompt: Union[str, List[str]], enable_positive_prompt=False):
    if isinstance(prompt, str):
        prompt = [prompt]
    
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    
    print(f'prompt={prompt}')
    # captions = [prompt]
    captions = prompt
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)    
    kv_compact = []
    
    for len_i, feat_i in zip(lens, text_features.unbind(0)):
        kv_compact.append(feat_i[:len_i])
    kv_compact = torch.cat(kv_compact, dim=0)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    
    return text_cond_tuple


def encode_video_prompt(t5_path, text_tokenizer, text_encoder, prompt, enable_positive_prompt=False, low_vram_mode=False):
    if enable_positive_prompt:
        pass
    print(f'prompt={prompt}')
    captions = [prompt]
    if 'flan-t5' in t5_path:
        tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.cuda(non_blocking=True)
        mask = tokens.attention_mask.cuda(non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)    
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    else:
        text_features = text_encoder(captions, 'cuda')
        lens = [len(item) for item in text_features]
        cu_seqlens_k = [0]
        for len_i in lens:
            cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
        Ltext = max(lens)
        kv_compact = torch.cat(text_features, dim=0).float()
        text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple


def aug_with_positive_prompt(prompt):
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def gen_one_img(
    infinity_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,     # = [0]
    vae_type=0,                 # = 32
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,                 # = 0
    gt_ls_Bl=None,
    g_seed=None,                # e.g. 2343
    sampling_per_bits=1,
    enable_positive_prompt=0,
    batch=1,
    test_speed=False,           # evaluate benchmark average latency
    **kwargs,
):
    # print(f'in gen_one_img: {g_seed=}')     # None
    # for sparse attn layer count
    from models.sparvar.sparse_attn_layer_counter import singleton as layer_counter

    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)   # ()
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    Infinity.autoregressive_infer_cfg   # for debug
    with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
        stt = time.time()
        _, _, img_list = infinity_test.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=batch, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            **kwargs,
        )
    cost, infinity_cost = time.time() - sstt, time.time() - stt
    # print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    print(f"cost: {cost}, infinity cost={infinity_cost}")
    
    if batch == 1: img = img_list[0]
    else: pass      # TODO: batch test

    # for sparse attn layer count
    layer_counter.reset()

    if test_speed:
        return img, cost, infinity_cost
    
    return img


def gen_one_video(
    infinitystar_test, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
    input_use_interplote_up=False,
    low_vram_mode=False,
    args=None,
    get_visual_rope_embeds=None,
    context_info=None,
    noise_list=None,
    return_summed_code_only=False,
    mode='',
    former_clip_features=None,
    first_frame_features=None,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    text_cond_tuple = encode_video_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, prompt, enable_positive_prompt, low_vram_mode=low_vram_mode)
    if negative_prompt:
        negative_label_B_or_BLT = encode_video_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, negative_prompt, low_vram_mode=low_vram_mode)
    else:
        negative_label_B_or_BLT = None
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    
    with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        out = infinitystar_test.autoregressive_infer(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            input_use_interplote_up=input_use_interplote_up,
            low_vram_mode=low_vram_mode,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            context_info=context_info,
            noise_list=noise_list,
            return_summed_code_only=return_summed_code_only,
            mode=mode,
            former_clip_features=former_clip_features,
            first_frame_features=first_frame_features,
        )
        if return_summed_code_only:
            return out
        else:
            pred_multi_scale_bit_labels, img_list = out
            
    print(f"cost: {time.time() - sstt}, infinity cost={time.time() - stt}")
    img = img_list[0]
    return img, pred_multi_scale_bit_labels


def save_video(ndarray_image_list, fps=24, save_filepath='tmp.mp4'):
    if len(ndarray_image_list) == 1:
        save_filepath = save_filepath.replace('.mp4', '.jpg')
        cv2.imwrite(save_filepath, ndarray_image_list[0])
        print(f"Image saved as {osp.abspath(save_filepath)}")
    else:
        h, w = ndarray_image_list[0].shape[:2]
        os.makedirs(osp.dirname(save_filepath), exist_ok=True)
        imageio.mimsave(save_filepath, ndarray_image_list[:, :, :, ::-1], fps=fps,)
        print(f"Video saved as {osp.abspath(save_filepath)}")


def get_prompt_id(prompt):
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    
    return prompt_id


def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device)
    infinity_slim = full_ckpt['trainer'][key]
    # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')

    return save_file


def load_tokenizer(t5_path='', load_weights=True):
    print(f'[Loading tokenizer and text encoder (load_weights={load_weights})]')

    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    # text_tokenizer.model_max_length = 512     # infinitystar no need

    if load_weights:
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, dtype=torch.float16)
        text_encoder.to('cuda')
    else:
        config = T5Config.from_pretrained(t5_path, dtype=torch.float16)
        text_encoder = T5EncoderModel(config).to('cuda')

    text_encoder.eval()
    text_encoder.requires_grad_(False)

    return text_tokenizer, text_encoder


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
    args=None,
    # todo: --- exp params ---
    # freeze_kv_cache_last_n_scales: int = 4
    attn_sink_scales: int = 5,
    skip_last_scales: int = 0,
    drop_uncond_last_scales: int = 3,
):
    text_maxlen = 512
    print(f'[Loading Infinity]')
    with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True), torch.no_grad():
        if model_kwargs.pop('fastvar', None) is not None:
            # print(f'{args.prune_ratio.split(',')=}')
            prune_ratio = tuple([float(item) for item in args.prune_ratio.split(',')])
            infinity_test: Infinity = FastVAR_Infinity(
                vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
                shared_aln=True, raw_scale_schedule=scale_schedule,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=use_flex_attn,
                add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
                use_bit_label=use_bit_label,
                rope2d_each_sa_layer=rope2d_each_sa_layer,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                pn=pn,
                apply_spatial_patchify=apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=[1.0],
                skip_last_scales=skip_last_scales,
                # pruning setting
                cached_scale=args.cached_scale,
                prune_ratio=prune_ratio,
                **model_kwargs,
            ).to(device=device)
            print(f'\n[you selected FastVAR with {model_kwargs=}] \
                  model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')
        
        elif model_kwargs.pop('skipvar', None) is not None:
            infinity_test: Infinity = SkipVAR_Infinity(
                vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
                shared_aln=True, raw_scale_schedule=scale_schedule,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=use_flex_attn,
                add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
                use_bit_label=use_bit_label,
                rope2d_each_sa_layer=rope2d_each_sa_layer,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                pn=pn,
                apply_spatial_patchify=apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=[1.0],
                **model_kwargs,
            ).to(device=device)
            print(f'\n[you selected SkipVAR with {model_kwargs=}] \
                  model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')
        
        elif model_kwargs.pop('sparsevar', None) is not None:
            infinity_test: Infinity = SparseVAR_Infinity(
                vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
                shared_aln=True, raw_scale_schedule=scale_schedule,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=use_flex_attn,
                add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
                use_bit_label=use_bit_label,
                rope2d_each_sa_layer=rope2d_each_sa_layer,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                pn=pn,
                apply_spatial_patchify=apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=[1.0],
                **model_kwargs,
            ).to(device=device)
            print(f'\n[you selected SparseVAR with {model_kwargs=}] \
                  model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')
        
        elif model_kwargs.pop('sparvar', None) is not None:
            infinity_test: SparVAR_Infinity = SparVAR_Infinity(
                vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
                shared_aln=True, raw_scale_schedule=scale_schedule,
                checkpointing='full-block',
                # customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=use_flex_attn,
                add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
                use_bit_label=use_bit_label,
                rope2d_each_sa_layer=rope2d_each_sa_layer,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                pn=pn,
                apply_spatial_patchify=apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=[1.0],
                skip_last_scales=skip_last_scales,
                drop_uncond_last_scales=drop_uncond_last_scales,
                **model_kwargs,
            ).to(device=device)
            print(f'\n[you selected SparVAR with {model_kwargs=}] \
                  model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')
        
        else:
            infinity_test: Infinity = Infinity(
                vae_local=vae, text_channels=text_channels, text_maxlen=text_maxlen,
                shared_aln=True, raw_scale_schedule=scale_schedule,
                checkpointing='full-block',
                # customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=use_flex_attn,
                add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
                use_bit_label=use_bit_label,
                rope2d_each_sa_layer=rope2d_each_sa_layer,
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                pn=pn,
                apply_spatial_patchify=apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=[1.0],
                # todo: --- exp params ---
                # freeze_kv_cache_last_n_scales=freeze_kv_cache_last_n_scales,
                # base_cache_scales=base_cache_scales,
                skip_last_scales=skip_last_scales,
                attn_sink_scales=attn_sink_scales,
                drop_uncond_last_scales=drop_uncond_last_scales,
                **model_kwargs,
            ).to(device=device)
            print(f'\n[you selected Infinity with {model_kwargs=}] \
                  model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch' and model_path is not None:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard' and model_path is not None:
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        
        return infinity_test


def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    # 1. Image load and pre-process
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)

    # 2. VAE encoder
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    t1 = time.time()
    h, z, _, all_bit_indices, _, infinity_input = vae.encode(inp, scale_schedule=scale_schedule)
    t2 = time.time()

    # 3. Call the VAE decoder for reconstruction
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    
    # 4. Image post-process
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    t3 = time.time()
    print(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    gt_img = (inp[0] + 1) / 2
    gt_img = gt_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)

    return gt_img, recons_img, all_bit_indices


def load_visual_tokenizer(args, load_weights=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [14, 16, 18, 20, 24, 32, 64]:
        from models.vae.bsq_vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]

        vae_path = args.vae_path if load_weights else None
        vae = vae_model(vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size,
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae


def load_transformer(vae, args, load_weights=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path if load_weights else None

    if args.checkpoint_type == 'torch':
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
        
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path
    
    if args.model_type == 'infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    elif args.model_type == 'infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    # elif 'infinity_2b' in args.model_type:
    #     kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    # --- add FastVAR (ICCV'2025) ---
    elif args.model_type == 'fastvar_infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            fastvar=True)
    elif args.model_type == 'fastvar_infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8, 
                            fastvar=True)
    # --- add ScaleKV (NeurIPS'2025) ---
    elif args.model_type == 'scalekv_infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'scalekv_infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    # --- add SkipVAR ---
    elif args.model_type == 'skipvar_infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            skipvar=True)
    elif args.model_type == 'skipvar_infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            skipvar=True)
    # --- add SparseVAR1 (ICCV'2025) ---
    elif args.model_type == 'sparsevar_infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            sparsevar1=True)
    elif args.model_type == 'sparsevar_infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            sparsevar1=True)
    # --- add SparseVAR ---
    elif args.model_type == 'sparvar_infinity_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            sparsevar=True)
    elif args.model_type == 'sparvar_infinity_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
                            sparsevar=True)
    
    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
        args=args,
        # todo: --- exp params ---
        # freeze_kv_cache_last_n_scales=args.freeze_kv_cache_last_n_scales
        attn_sink_scales=args.attn_sink_scales,
        skip_last_scales=args.skip_last_scales,
        drop_uncond_last_scales=args.drop_uncond_last_scales
    )

    if 'scalekv' in args.model_type:
        infinity = enable_scale_kv(infinity, window_size=16, max_capacity=650,
                                   kernel_size=5, pooling='maxpool',
                                   model_size=args.model_type.split('_')[-1])

    return infinity


def load_video_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    
    print(f'[Loading InfinityStar]')
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity = create_model(
            args.model_type,
            vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
            raw_scale_schedule=None,
            checkpointing='full-block',
            pad_to_multiplier=128,
            use_flex_attn=args.use_flex_attn,
            add_lvl_embeding_on_first_block=0,
            num_of_label_value=args.num_of_label_value,
            rope2d_each_sa_layer=args.rope2d_each_sa_layer,
            rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
            pn=args.pn,
            apply_spatial_patchify=args.apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[0.571, 1.0],
            video_frames=args.video_frames,
            other_args=args,
        ).to(device=device)
        print(f'[you selected Infinity with {args.model_type}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={args.bf16}')
        if args.bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()
        infinity_test.eval()
        infinity_test.requires_grad_(False)
        infinity_test.cuda()
        torch.cuda.empty_cache()

        if not model_path:
            return infinity_test
        
        print(f'============== [Load Infinity weights] ==============')    
        if args.checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            print(infinity_test.load_state_dict(state_dict))
        elif args.checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            print(load_sharded_checkpoint(infinity_test, model_path, strict=False))
        elif args.checkpoint_type == 'omnistore':
            from utils.save_and_load import merge_ckpt
            if args.enable_model_cache and osp.exists(args.cache_dir):
                local_model_dir = osp.abspath(osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_')))
            else:
                local_model_dir = osp.abspath(model_path)
            print(f'load checkpoint from {local_model_dir}')
            state_dict = merge_ckpt(local_model_dir, osp.join(local_model_dir, 'ouput'), save=False, fsdp_save_flatten_model=args.fsdp_save_flatten_model)
            print(infinity_test.load_state_dict(state_dict))
            import pdb; pdb.set_trace()
            # # split_state_dict
            # save_directory = '/tmp/weights/infinity_interact_24k'
            # os.makedirs(save_directory, exist_ok=True)
            # split_state_dict(state_dict, save_directory)
        infinity_test.rng = torch.Generator(device=device)
    
    return infinity_test


def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, default='1M', choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, default='pretrained_models/infinity/Infinity/infinity_2b_reg.pth')
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=32)
    parser.add_argument('--vae_path', type=str, default='pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='pretrained_models/infinity/flan-t5-xl')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    # ------ FastVAR ------
    parser.add_argument('--cached_scale', type=int, default=8)
    # parser.add_argument("--prune_ratio", nargs='+', type=float, default=[0.4, 0.5], help="Pruning ratio for last 2 scales in FastVAR")
    parser.add_argument("--prune_ratio", type=str, default="0.4,0.5", help="Pruning ratio for last 2 scales in FastVAR")
    # ------ SparseVAR ------
    parser.add_argument("--compress_method", type=str, help="compress method", default="sparsevar")
    parser.add_argument("--compress_ratio", type=float, help="compress ratio", default=0.6)
    # parser.add_argument("--test_speed", action="store_true", default=False)
    parser.add_argument("--start_prune_stage", type=int, default=10)
    parser.add_argument("--specific_mse_layer", type=int, default=3)
    parser.add_argument("--local_window_size", type=int, default=4)
    # ------ HART ------
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument("--more_smooth", type=bool, help="Turn on for more visually smooth samples.", default=True)
    # ------ exp params ------
    # parser.add_argument('--freeze_kv_cache_last_n_scales', type=int, default=4)
    parser.add_argument('--attn_sink_scales', type=int, default=5, help='Sink the attention maps of the last few scales')
    parser.add_argument('--skip_last_scales', type=int, default=0, help='Skip the last few scales')
    parser.add_argument('--drop_uncond_last_scales', type=int, default=0, help='Drop the unconditional branch of last few scales')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./tmp.jpg')
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    with autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img(
                infinity,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
            )
    
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')
