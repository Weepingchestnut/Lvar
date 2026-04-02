import argparse
# import copy
import os
import random
import time
from typing import Dict, Sequence

import numpy as np
import pynvml
import torch
import torchvision
import transformers
from PIL import Image
from tqdm import tqdm
# from transformers import AutoModel, AutoTokenizer

from tools.run_hart import llm_system_prompt, load_hart, load_hart_tokenizer
from tools.run_infinity import add_common_arguments
from utils.profile_utils import (default_prompts, format_memory,
                                 get_memory_usage, get_physical_device_index)


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
    # --- NVML init ---
    nvml_handle = None
    try:
        pynvml.nvmlInit()
        if torch.cuda.is_available():
            # Use the helper function to get the correct physical device index
            physical_device_index = get_physical_device_index()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_index)
            print(f"PyTorch is using device: cuda:{torch.cuda.current_device()}. Monitoring physical NVIDIA GPU index: {physical_device_index}")
    except pynvml.NVMLError as e:
        print(f"Warning: Could not initialize NVML. NVIDIA-SMI usage will not be reported. Error: {e}")
    # ----------------
    lib = torch.cuda.cudart()

    try:
        device = torch.device("cuda")
    
        # --- Memory Analysis: Initial State ---
        torch.cuda.empty_cache()
        start_cpu, start_pytorch_gpu, start_nvsmi_gpu = get_memory_usage(device, nvml_handle)
        print("-------- Memory Usage (Initial) --------")
        print(f"CPU Memory: {format_memory(start_cpu)}")
        print(f"GPU Memory (PyTorch): {format_memory(start_pytorch_gpu)}")
        print(f"GPU Memory (NVIDIA-SMI): {format_memory(start_nvsmi_gpu)}")
        print("-" * 40)
        # --------------------------------------
        
        text_tokenizer, text_model = load_hart_tokenizer(args, device)
        model, ema_model = load_hart(args, device)

        # --- Memory analysis: After model load ---
        torch.cuda.synchronize()
        post_cpu, post_pytorch_gpu, post_nvsmi_gpu = get_memory_usage(device, nvml_handle)
        print("\n--- Memory Usage (After Model Load) ---")
        print(f"CPU Memory: {format_memory(post_cpu)} (Used: {format_memory(post_cpu - start_cpu)})")
        print(f"GPU Memory (PyTorch): {format_memory(post_pytorch_gpu)} (Used: {format_memory(post_pytorch_gpu - start_pytorch_gpu)})")
        print(f"GPU Memory (NVIDIA-SMI): {format_memory(post_nvsmi_gpu)} (Used: {format_memory(post_nvsmi_gpu - start_nvsmi_gpu)})")
        print("-" * 35)
        # ----------------------------------------

        seed = args.seed
        text_tokenizer_max_length = args.max_token_length
        # prompts = random.sample(default_prompts, args.batch_size)

        with torch.inference_mode():
            with torch.autocast(
                "cuda", enabled=True, 
                dtype=torch.float16, 
                cache_enabled=True
            ):

                infer_func = (
                    ema_model.autoregressive_infer_cfg
                    if args.use_ema
                    else model.autoregressive_infer_cfg
                )

                # --- Memory Analysis: Preparing for Inference ---
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                peak_cpu_mem = post_cpu
                peak_nvsmi_gpu_mem = post_nvsmi_gpu
                # ------------------------------------------------

                # warmup
                print(f"\nStarting GPU warm-up for {args.warmup_iter} iterations...")
                for _ in tqdm(range(args.warmup_iter)):
                    prompts = random.sample(default_prompts, args.batch_size)
                    (
                        context_tokens,
                        context_mask,
                        context_position_ids,
                        context_tensor,
                    ) = encode_prompts(
                        prompts,
                        text_model,
                        text_tokenizer,
                        text_tokenizer_max_length,
                        llm_system_prompt,
                        args.use_llm_system_prompt,
                    )
                    lib.cudaProfilerStart()
                    output_imgs = infer_func(
                        B=context_tensor.size(0),
                        label_B=context_tensor,
                        cfg=args.cfg,
                        g_seed=args.seed,
                        more_smooth=args.more_smooth,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                    )
                    lib.cudaProfilerStop()
                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------
                torch.cuda.synchronize(device=device)
                print("GPU warm-up finished.")

                # latency profile
                print(f"\nStarting inference speed test for {args.profile_iter} iterations...")
                timings = []
                sstart_time = time.time()
                for _ in tqdm(range(args.profile_iter)):
                    prompts = random.sample(default_prompts, args.batch_size)
                    # print('\n====== prompts ======')
                    # print(f'{prompts}')
                    # print(f'=====================')
                    
                    (
                        context_tokens,
                        context_mask,
                        context_position_ids,
                        context_tensor,
                    ) = encode_prompts(
                        prompts,
                        text_model,
                        text_tokenizer,
                        text_tokenizer_max_length,
                        llm_system_prompt,
                        args.use_llm_system_prompt,
                    )

                    start_time = time.perf_counter()    # for accurate timing
                    lib.cudaProfilerStart()
                    output_imgs = infer_func(
                        B=context_tensor.size(0),
                        label_B=context_tensor,
                        cfg=args.cfg,
                        g_seed=args.seed,
                        more_smooth=args.more_smooth,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                    )
                    torch.cuda.synchronize(device=device)   # *Important*: Ensure that all CUDA operations are completed before recording the time
                    lib.cudaProfilerStop()

                    end_time = time.perf_counter()
                    timings.append(end_time - start_time); # print(f'%%%%%% {(end_time - start_time)/args.batch_size * 1000:.2f}ms %%%%%%')

                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------
                    # print(f'{type(output_imgs)=}, {output_imgs.shape=}, {output_imgs=}')

                ttotal_time = time.time() - sstart_time
                print("Inference speed test finished.")

        average_time = ttotal_time / args.profile_iter
        print(f"\nGeneration with batch_size = {args.batch_size} take {average_time:2f}s per step (w/ text encoder).")

        batch_size = args.batch_size
        avg_batch_latency = sum(timings) / len(timings)
        std_batch_latency = torch.tensor(timings).std().item()
        throughput = batch_size / avg_batch_latency if avg_batch_latency > 0 else float('inf')
        # per-image latency
        avg_latency = avg_batch_latency / batch_size
        std_latency = std_batch_latency / batch_size

        print(f"\n--- Inference Performance ---")
        print(f"Batch Size: {batch_size}")
        print(f"Average Latency: {avg_latency * 1000:.2f} ms")
        print(f"Latency StdDev: {std_latency * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print("-" * 30)

        # --- Memory Analysis: final report ---
        peak_pytorch_gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print("\n----------- Memory Usage (Peak during Inference) -----------")
        print(f"Peak CPU Memory: {format_memory(peak_cpu_mem)} (Total Used: {format_memory(peak_cpu_mem - start_cpu)})")
        print(f"Peak GPU Memory (PyTorch): {format_memory(peak_pytorch_gpu_mem)} (Total Used: {format_memory(peak_pytorch_gpu_mem - start_pytorch_gpu)})")
        print(f"Peak GPU Memory (NVIDIA-SMI): {format_memory(peak_nvsmi_gpu_mem)} (Total Used: {format_memory(peak_nvsmi_gpu_mem - start_nvsmi_gpu)})")
        print("-" * 60)
    
    finally:
        if nvml_handle:
            pynvml.nvmlShutdown()   # NVML clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    parser.add_argument(
        "--batch_size", type=int, help="Generation batch size", default=1)
    parser.add_argument("--warmup_iter", type=int, default=50)
    parser.add_argument("--profile_iter", type=int, default=100)
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    main(args)
