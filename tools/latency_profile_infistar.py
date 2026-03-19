import argparse
import os.path as osp
import random
import sys
import time
from typing import List, Union

import numpy as np
import pynvml
import torch
from tqdm import tqdm

# Make sure repo root is importable no matter where the script is launched from.
sys.path.append(osp.dirname(osp.dirname(__file__)))

# Ensure InfinityStar models are registered into timm before create_model().
import models.infinitystar  # noqa: F401
from models.schedules import get_encode_decode_func
from models.schedules.dynamic_resolution import (get_dynamic_resolution_meta,
                                                 get_first_full_spatial_size_scale_index)
from run_infinity import gen_one_video, load_tokenizer, load_video_transformer
from utils.arg_util_video import Args
from utils.load import load_video_visual_tokenizer
from utils.profile_utils import (default_prompts, format_memory, get_memory_usage,
                                 get_physical_device_index)


def parse_cfg(raw_cfg: str) -> Union[float, List[float]]:
    try:
        cfg = list(map(float, raw_cfg.split(",")))
    except ValueError as e:
        raise ValueError(f"Invalid --cfg value: {raw_cfg!r}. Expected a float or comma-separated floats.") from e
    if len(cfg) == 0:
        raise ValueError("--cfg must not be empty.")
    return cfg[0] if len(cfg) == 1 else cfg


def format_duration_seconds(duration_seconds: float) -> str:
    if abs(duration_seconds - round(duration_seconds)) < 1e-6:
        return str(int(round(duration_seconds)))
    return f"{duration_seconds:.2f}"


def build_video_args(cli_args: argparse.Namespace, cfg: Union[float, List[float]]) -> Args:
    args = Args()
    args.pn = cli_args.pn
    args.fps = cli_args.fps
    args.video_frames = cli_args.video_frames
    args.model_path = cli_args.model_path
    args.checkpoint_type = cli_args.checkpoint_type
    args.vae_path = cli_args.vae_path
    args.text_encoder_ckpt = cli_args.text_encoder_ckpt
    args.videovae = cli_args.videovae
    args.model_type = cli_args.model_type
    args.text_channels = cli_args.text_channels
    args.dynamic_scale_schedule = cli_args.dynamic_scale_schedule
    args.bf16 = cli_args.bf16
    args.use_apg = cli_args.use_apg
    args.use_cfg = cli_args.use_cfg
    args.cfg = cfg
    args.tau_image = cli_args.tau_image
    args.tau_video = cli_args.tau_video
    args.apg_norm_threshold = cli_args.apg_norm_threshold
    args.image_scale_repetition = cli_args.image_scale_repetition
    args.video_scale_repetition = cli_args.video_scale_repetition
    args.append_duration2caption = cli_args.append_duration2caption
    args.use_two_stage_lfq = cli_args.use_two_stage_lfq
    args.detail_scale_min_tokens = cli_args.detail_scale_min_tokens
    args.semantic_scales = cli_args.semantic_scales
    args.max_repeat_times = cli_args.max_repeat_times
    args.vae_type = cli_args.vae_type
    args.use_flex_attn = cli_args.use_flex_attn
    args.apply_spatial_patchify = cli_args.apply_spatial_patchify
    args.num_of_label_value = cli_args.num_of_label_value
    return args


def build_scale_context(
    args: Args,
    h_div_w_template: float,
):
    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
        args.dynamic_scale_schedule,
        args.video_frames,
    )
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w_template))]
    if args.pn not in dynamic_resolution_h_w[h_div_w_template_]:
        valid_pn = sorted(dynamic_resolution_h_w[h_div_w_template_].keys())
        raise ValueError(f"--pn={args.pn!r} is not available for h/w template {h_div_w_template_}. Valid values: {valid_pn}")
    schedule_idx = (args.video_frames - 1) // 4 + 1
    scale_dict = dynamic_resolution_h_w[h_div_w_template_][args.pn]["pt2scale_schedule"]
    if schedule_idx not in scale_dict:
        valid_frames = sorted(((k - 1) * 4 + 1) for k in scale_dict.keys())
        raise ValueError(
            f"--video_frames={args.video_frames} is not available for current dynamic schedule. "
            f"Valid frame counts include: {valid_frames[:8]}{'...' if len(valid_frames) > 8 else ''}"
        )
    scale_schedule = scale_dict[schedule_idx]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1

    _, _, get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
    context_info = get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)
    tau_list = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    return scale_schedule, tau_list, get_visual_rope_embeds, context_info


def main(cli_args):
    if cli_args.batch_size != 1:
        raise ValueError("InfinityStar benchmark currently supports only --batch_size=1.")
    if cli_args.use_apg + cli_args.use_cfg != 1:
        raise ValueError("Exactly one of --use_apg or --use_cfg must be 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency profiling.")

    random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    cfg = parse_cfg(cli_args.cfg)
    args = build_video_args(cli_args, cfg)
    if cli_args.video_frames < 5 or (cli_args.video_frames - 1) % 4 != 0:
        raise ValueError("--video_frames must satisfy video_frames >= 5 and (video_frames - 1) % 4 == 0.")
    scale_schedule, tau_list, get_visual_rope_embeds, context_info = build_scale_context(
        args,
        cli_args.h_div_w_template,
    )
    if isinstance(cfg, list) and len(cfg) not in {1} and len(cfg) < len(scale_schedule):
        raise ValueError(
            f"--cfg list length ({len(cfg)}) must be 1 or >= number of scales ({len(scale_schedule)})."
        )

    mapped_duration = max((args.video_frames - 1) / cli_args.fps, 0.0)
    mapped_duration_str = format_duration_seconds(mapped_duration)

    # --- NVML init ---
    nvml_handle = None
    try:
        pynvml.nvmlInit()
        physical_device_index = get_physical_device_index()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_index)
        print(
            f"PyTorch is using device: cuda:{torch.cuda.current_device()}. "
            f"Monitoring physical NVIDIA GPU index: {physical_device_index}"
        )
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

        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        vae = load_video_visual_tokenizer(args).float().to("cuda")
        infinitystar = load_video_transformer(vae, args)

        # --- Memory analysis: After model load ---
        torch.cuda.synchronize()
        post_cpu, post_pytorch_gpu, post_nvsmi_gpu = get_memory_usage(device, nvml_handle)
        print("\n--- Memory Usage (After Model Load) ---")
        print(f"CPU Memory: {format_memory(post_cpu)} (Used: {format_memory(post_cpu - start_cpu)})")
        print(
            f"GPU Memory (PyTorch): {format_memory(post_pytorch_gpu)} "
            f"(Used: {format_memory(post_pytorch_gpu - start_pytorch_gpu)})"
        )
        print(
            f"GPU Memory (NVIDIA-SMI): {format_memory(post_nvsmi_gpu)} "
            f"(Used: {format_memory(post_nvsmi_gpu - start_nvsmi_gpu)})"
        )
        print("-" * 35)
        # ----------------------------------------

        with torch.inference_mode():
            # --- Memory Analysis: Preparing for Inference ---
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            peak_cpu_mem = post_cpu
            peak_nvsmi_gpu_mem = post_nvsmi_gpu
            # ------------------------------------------------

            print(f"\nStarting GPU warm-up for {cli_args.warmup_iter} iterations...")
            for _ in tqdm(range(cli_args.warmup_iter)):
                prompt = random.choice(default_prompts)
                if cli_args.prompt_suffix:
                    prompt = f"{prompt}, {cli_args.prompt_suffix}"
                if args.append_duration2caption:
                    prompt = f"<<<t={mapped_duration_str}s>>>{prompt}"

                lib.cudaProfilerStart()
                gen_one_video(
                    infinitystar,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt,
                    negative_prompt=cli_args.negative_prompt,
                    g_seed=cli_args.seed,
                    cfg_list=cfg,
                    tau_list=tau_list,
                    scale_schedule=scale_schedule,
                    top_k=cli_args.top_k,
                    top_p=cli_args.top_p,
                    cfg_insertion_layer=[cli_args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=cli_args.sampling_per_bits,
                    enable_positive_prompt=0,
                    low_vram_mode=bool(cli_args.low_vram_mode),
                    args=args,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    context_info=context_info,
                    noise_list=None,
                )
                lib.cudaProfilerStop()

                cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                peak_cpu_mem = max(peak_cpu_mem, cpu)
                peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
            torch.cuda.synchronize(device=device)
            print("GPU warm-up finished.")

            print(f"\nStarting inference speed test for {cli_args.profile_iter} iterations...")
            timings = []
            sstart_time = time.time()
            for _ in tqdm(range(cli_args.profile_iter)):
                prompt = random.choice(default_prompts)
                if cli_args.prompt_suffix:
                    prompt = f"{prompt}, {cli_args.prompt_suffix}"
                if args.append_duration2caption:
                    prompt = f"<<<t={mapped_duration_str}s>>>{prompt}"

                start_time = time.perf_counter()
                lib.cudaProfilerStart()
                gen_one_video(
                    infinitystar,
                    vae,
                    text_tokenizer,
                    text_encoder,
                    prompt,
                    negative_prompt=cli_args.negative_prompt,
                    g_seed=cli_args.seed,
                    cfg_list=cfg,
                    tau_list=tau_list,
                    scale_schedule=scale_schedule,
                    top_k=cli_args.top_k,
                    top_p=cli_args.top_p,
                    cfg_insertion_layer=[cli_args.cfg_insertion_layer],
                    vae_type=args.vae_type,
                    sampling_per_bits=cli_args.sampling_per_bits,
                    enable_positive_prompt=0,
                    low_vram_mode=bool(cli_args.low_vram_mode),
                    args=args,
                    get_visual_rope_embeds=get_visual_rope_embeds,
                    context_info=context_info,
                    noise_list=None,
                )
                torch.cuda.synchronize(device=device)
                lib.cudaProfilerStop()
                end_time = time.perf_counter()

                timings.append(end_time - start_time)
                cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                peak_cpu_mem = max(peak_cpu_mem, cpu)
                peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)

            ttotal_time = time.time() - sstart_time
            print("Inference speed test finished.")

        average_time = ttotal_time / cli_args.profile_iter
        print(f"\nGeneration with batch_size = 1 take {average_time:2f}s per step (w/ text encoder).")

        avg_batch_latency = sum(timings) / len(timings)
        std_batch_latency = torch.tensor(timings).std().item()
        throughput = 1.0 / avg_batch_latency if avg_batch_latency > 0 else float("inf")

        print(f"\n--- Inference Performance ---")
        print("Batch Size: 1")
        print(f"Average Latency: {avg_batch_latency * 1000:.2f} ms")
        print(f"Latency StdDev: {std_batch_latency * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print("-" * 30)

        # --- Memory Analysis: final report ---
        peak_pytorch_gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print("\n----------- Memory Usage (Peak during Inference) -----------")
        print(f"Peak CPU Memory: {format_memory(peak_cpu_mem)} (Total Used: {format_memory(peak_cpu_mem - start_cpu)})")
        print(
            f"Peak GPU Memory (PyTorch): {format_memory(peak_pytorch_gpu_mem)} "
            f"(Total Used: {format_memory(peak_pytorch_gpu_mem - start_pytorch_gpu)})"
        )
        print(
            f"Peak GPU Memory (NVIDIA-SMI): {format_memory(peak_nvsmi_gpu_mem)} "
            f"(Total Used: {format_memory(peak_nvsmi_gpu_mem - start_nvsmi_gpu)})"
        )
        print("-" * 60)
    finally:
        if nvml_handle:
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_iter", type=int, default=50)
    parser.add_argument("--profile_iter", type=int, default=100)

    parser.add_argument("--cfg", type=str, default="34")
    parser.add_argument("--use_apg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_cfg", type=int, default=0, choices=[0, 1])
    parser.add_argument("--apg_norm_threshold", type=float, default=0.05)
    parser.add_argument("--tau_image", type=float, default=1.0)
    parser.add_argument("--tau_video", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.97)

    parser.add_argument("--pn", type=str, default="0.90M")
    parser.add_argument("--h_div_w_template", type=float, default=0.571)
    parser.add_argument("--video_frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dynamic_scale_schedule", type=str, default="infinity_elegant_clip20frames_v2")

    parser.add_argument("--model_type", type=str, default="infinitystar_qwen8b")
    parser.add_argument("--model_path", type=str, default="pretrained_models/infinitystar/infinitystar_8b_720p_weights")
    parser.add_argument("--checkpoint_type", type=str, default="torch_shard")
    parser.add_argument("--vae_path", type=str, default="pretrained_models/infinitystar/infinitystar_videovae.pth")
    parser.add_argument("--text_encoder_ckpt", type=str, default="pretrained_models/infinitystar/text_encoder/flan-t5-xl-official/")
    parser.add_argument("--videovae", type=int, default=10)
    parser.add_argument("--vae_type", type=int, default=64)
    parser.add_argument("--text_channels", type=int, default=2048)
    parser.add_argument("--num_of_label_value", type=int, default=2)
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_flex_attn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--apply_spatial_patchify", type=int, default=0, choices=[0, 1])

    parser.add_argument("--image_scale_repetition", type=str, default="[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]")
    parser.add_argument("--video_scale_repetition", type=str, default="[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]")
    parser.add_argument("--append_duration2caption", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_two_stage_lfq", type=int, default=1, choices=[0, 1])
    parser.add_argument("--detail_scale_min_tokens", type=int, default=750)
    parser.add_argument("--semantic_scales", type=int, default=12)
    parser.add_argument("--max_repeat_times", type=int, default=10000)

    parser.add_argument("--cfg_insertion_layer", type=int, default=0)
    parser.add_argument("--sampling_per_bits", type=int, default=1, choices=[1, 2, 4, 8, 16])
    parser.add_argument("--low_vram_mode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="Close-up on big objects, emphasize scale and detail",
    )

    main(parser.parse_args())
