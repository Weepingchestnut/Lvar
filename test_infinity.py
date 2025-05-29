import random
import torch

import cv2
import numpy as np
from tools.run_infinity import *

# --- infinity 2b ---
model_type = 'infinity_2b'
vae_type = 32
apply_spatial_patchify = 0
checkpoint_type = 'torch'
tau = 0.5

model_path = "pretrained_models/infinity/Infinity/infinity_2b_reg.pth"
vae_path = 'pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth'

# --- infinity 8b ---
# model_type = 'infinity_8b'
# vae_type=14
# apply_spatial_patchify=1
# checkpoint_type='torch_shard'
# tau = 1

# model_path = "pretrained_models/infinity/Infinity/infinity_8b_weights"
# vae_path = 'pretrained_models/infinity/Infinity/infinity_vae_d56_f8_14_patchify.pth'

text_encoder_ckpt = 'pretrained_models/infinity/flan-t5-xl'

args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=vae_type,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type=model_type,
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=apply_spatial_patchify,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type=checkpoint_type,
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    # 
    enable_model_cache=0
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

# --- prompt ---
# prompt = """alien spaceship enterprise"""
# prompt = """a cat holds a board with the text 'diffusion is dead'"""
# prompt = """A beautiful Chinese woman with graceful features, close-up portrait, long flowing black hair, wearing a traditional silk cheongsam delicately embroidered with floral patterns, face softly illuminated by ambient light, serene expression"""
# prompt = """a Chinese model is sitting on a train, magazine cover, clothes made of plastic, photorealistic, futuristic style, gray and green light, movie lighting, 32K HD"""
prompt = """A group of students in a class"""
# --------------
cfg = 3
# tau = 0.5     # different for Infinity 2b and 8b
h_div_w = 1/1 # aspect ratio, height:width
seed = random.randint(0, 10000)
enable_positive_prompt=0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
generated_image = gen_one_img(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    g_seed=seed,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=cfg,
    tau_list=tau,
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=enable_positive_prompt,
)
args.save_file = 'ipynb_tmp.jpg'
os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
cv2.imwrite(args.save_file, generated_image.cpu().numpy())
print(f'Save to {osp.abspath(args.save_file)}')
