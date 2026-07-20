import os
import cv2
import random

import numpy as np
import torch

from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img
from utils.arg_util import InfinityInferArgs
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from utils.misc import time_str


if __name__ == '__main__':
    # infer args
    args = InfinityInferArgs()
    args.model_type = 'sparsevar_infinity_2b'

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    # --- prompt ---
    prompt = """alien spaceship enterprise"""
    # prompt = """a cat holds a board with the text 'diffusion is dead'"""
    # prompt = """A beautiful Chinese woman with graceful features, close-up portrait, long flowing black hair, wearing a traditional silk cheongsam delicately embroidered with floral patterns, face softly illuminated by ambient light, serene expression"""
    # prompt = """a Chinese model is sitting on a train, magazine cover, clothes made of plastic, photorealistic, futuristic style, gray and green light, movie lighting, 32K HD"""
    # prompt = """A group of students in a class"""
    # --------------
    h_div_w = 1/1 # aspect ratio, height:width

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
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

    save_path = 'work_dir/play_models/sparsevar'
    img_path = os.path.join(save_path, f"{args.model_type}_example_{time_str()}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    cv2.imwrite(img_path, generated_image.cpu().numpy())
    print(f'\nSave to {os.path.abspath(img_path)}')
