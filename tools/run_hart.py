import os

import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from PIL import Image
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
import torchvision
from torch import autocast
import transformers

from models.hart.hart_transformer_t2i import HARTForT2I


# ============
# * constants
# ============
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

max_seq_len = 10240
max_batch_size = 16


# ===============
# * safety check
# ===============
safety_policy = """
    The prompt must not seek to generate harmful, abusive, hateful, sexually explicit, or otherwise inappropriate content targeting individuals or protected groups.
"""


def is_dangerous(tokenizer, model, user_prompt, threshold=0.5):
    prompt = f"""You are a policy expert trying to help determine whether a user
    prompt is in violation of the defined safety policies.

    <start_of_turn>
    Human Question: {user_prompt.strip()}
    <end_of_turn>

    Our safety principle is defined in the below:

    {safety_policy.strip()}

    Does the human question violate the above principle? Your answer must start
    with 'Yes' or 'No'. And then walk through step by step to be sure we answer
    correctly.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits

    # Extract the logits for the Yes and No tokens
    vocab = tokenizer.get_vocab()
    selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]

    # Convert these logits to a probability with softmax
    probabilities = F.softmax(selected_logits, dim=0)

    # Return probability of 'Yes'
    score = probabilities[0].item()

    return score > threshold


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


def gen_one_img_hart(
    hart_test,
    use_ema,
    hart_test_ema,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg: float = 4.5,
    max_token_length: int = 300,
    use_llm_system_prompt: bool = True,
    more_smooth: bool = True,
    g_seed=None,
    **kwargs,
):
    with torch.inference_mode():
        with autocast("cuda", dtype=torch.float16, enabled=True, cache_enabled=True):
            sstt = time.time()
            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                [prompt],
                text_encoder,
                text_tokenizer,
                max_token_length,
                llm_system_prompt,
                use_llm_system_prompt,
            )

            infer_func = (
                hart_test_ema.autoregressive_infer_cfg
                if use_ema
                else hart_test.autoregressive_infer_cfg
            )
            stt = time.time()
            output_imgs = infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=cfg,
                g_seed=g_seed,
                more_smooth=more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )
    
    print(f"cost: {time.time() - sstt}, hart cost={time.time() - stt}, text_encoder cost={(time.time() - sstt) - (time.time() - stt)}")

    sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    assert num_imgs == 1, f'Evaluation only need 1 image, but now have {num_imgs}'
    cur_img = sample_imgs_np[0]
    cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)

    return Image.fromarray(cur_img)
