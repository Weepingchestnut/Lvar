import argparse
import copy
import os
import random
import time
from typing import Dict, Sequence

import numpy as np
import torch
import torchvision

from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoModel, AutoTokenizer

from models.hart.hart_transformer_t2i import HARTForT2I     # important for transformers pkg register


default_prompts = [
    "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
    "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur."
    "Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.",
    "A 4k dslr image of a lemur wearing a red magician hat and a blue coat performing magic tricks with cards in a garden.",
    "A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid sytle of a vintage travel poster.",
    "Crocodile in a sweater.",
    "Luffy from ONEPIECE, handsome face, fantasy.",
    "3d digital art of an adorable ghost, glowing within, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background.",
    "an astronaut sitting in a diner, eating fries, cinematic, analog film",
    "Chinese architecture, ancient style,mountain, bird, lotus, pond, big tree, 4K Unity, octane rendering.",
]

llm_system_prompt = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation.

Examples:
- User Prompt: A cat sleeping -> A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.

Please generate only the enhanced description for the prompt below and DO NOT include any additional sentences. Start your response with "Enhanced Prompt:".

User Prompt:\n"""

# max_seq_len = 10240
# max_batch_size = 16


# Modified from VILA
def tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    padding_mode: str = "longest",
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding_mode,
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return input_ids


def encode_prompts(
    prompts,
    text_model,
    text_tokenizer,
    text_tokenizer_max_length,
    system_prompt=None,
    use_llm_system_prompt=False,
):
    device = text_model.device
    tokenized_prompts = tokenize_fn(
        prompts,
        tokenizer=text_tokenizer,
        max_length=text_tokenizer_max_length,
        padding_mode="max_length",
    )
    context_tokens = torch.stack(tokenized_prompts).to(device)
    context_mask = context_tokens != text_tokenizer.pad_token_id
    context_position_ids = torch.cumsum(context_mask, dim=1) - 1

    if not use_llm_system_prompt:
        context_tensor = text_model(
            context_tokens, attention_mask=context_mask, output_hidden_states=True
        ).hidden_states[-1]
    else:
        system_prompt_tokens = tokenize_fn(
            [system_prompt],
            tokenizer=text_tokenizer,
            max_length=text_tokenizer_max_length,
            padding_mode="longest",
        )
        system_prompt_tokens = system_prompt_tokens[0].to(context_tokens.device)
        system_prompt_tokens = system_prompt_tokens.unsqueeze(0)
        system_prompt_tokens = system_prompt_tokens.repeat(context_tokens.shape[0], 1)
        system_prompt_mask = torch.ones_like(context_mask)[
            :, : system_prompt_tokens.shape[1]
        ]
        # include system prompt when calculating embeddings
        # but only keep the embedding for original tokens
        context_tensor = text_model(
            torch.cat([system_prompt_tokens, context_tokens], dim=1),
            attention_mask=torch.cat(
                [
                    system_prompt_mask,
                    context_mask,
                ],
                dim=1,
            ),
            output_hidden_states=True,
        ).hidden_states[-1][:, system_prompt_tokens.shape[1] :]
    context_tensor = context_tensor.float()

    return (context_tokens, context_mask, context_position_ids, context_tensor)


def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


def main(args):
    device = torch.device("cuda")

    model = AutoModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(args.text_model_path).to(device)
    text_model.eval()
    text_tokenizer_max_length = args.max_token_length

    prompts = random.sample(default_prompts, args.batch_size)

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):

            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            infer_func = (
                ema_model.autoregressive_infer_cfg
                if args.use_ema
                else model.autoregressive_infer_cfg
            )

            # warmup
            for _ in tqdm(range(args.warmup_iter)):
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )

            # latency profile
            start_time = time.time()
            for _ in tqdm(range(args.profile_iter)):
                (
                    context_tokens,
                    context_mask,
                    context_position_ids,
                    context_tensor,
                ) = encode_prompts(
                    prompts,
                    text_model,
                    text_tokenizer,
                    args.max_token_length,
                    llm_system_prompt,
                    args.use_llm_system_prompt,
                )
                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=args.cfg,
                    g_seed=args.seed,
                    more_smooth=args.more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
            total_time = time.time() - start_time

    average_time = total_time / args.profile_iter
    print(
        f"Generation with batch_size = {args.batch_size} take {average_time:2f}s per step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="pretrained_models/HART-1024",)
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",)
    parser.add_argument(
        "--batch_size", type=int, help="Generation batch size", default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5)
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,)
    parser.add_argument("--warmup_iter", type=int, default=50)
    parser.add_argument("--profile_iter", type=int, default=100)
    args = parser.parse_args()

    main(args)
