import datetime
import hashlib
import json
import os
import os.path as osp
import time
from typing import List

import numpy as np
import torch
import torch.distributed as tdist
from diffusers import AutoencoderKLWan, AutoModel, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import \
    UniPCMultistepScheduler
from diffusers.utils import export_to_video
from tqdm import tqdm
from transformers import UMT5EncoderModel

from utils.arg_util_video import Args


class InferArgs(Args):
    
    model_type: str = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    resolution: str = '480p'
    res_dict: dict = {
        '480p': {'height': 480, 'width': 832},
        '720p': {'height': 720, 'width': 1280}
    }
    fps: int = 16
    generation_duration: int = 5
    video_frames: int = generation_duration * fps + 1       # default 81 frames --> 5s video
    
    cfg: float = 5.0                # for WAN2.1
    diff_steps: int = 50
    if resolution == '480p':        # 5.0 for 720P, 3.0 for 480P
        flow_shift: float = 3.0
    else:
        flow_shift: float = 5.0
    
    # For optimal performance, enabling the prompt rewriter is recommended.
    # To utilize the GPT model, ensure the following environment variables are set:
    # export OPEN_API_KEY="YOUR_API_KEY"
    # export GLOBAL_AZURE_ENDPOINT="YOUR_ENDPOINT"
    # *--> use official rewrite VBench_rewrited_prompt.json
    enable_rewriter: int = 0
    
    # -------- VBench Setting --------
    prompt_json: str = 'evaluation/vbench/VBench_rewrited_prompt_fixed_seed.json'
    output_root: str = f'work_dir/evaluation/vbench/{model_type.split("/")[-1]}_480p_81frames'
    start_index: int = 0
    end_index: int = -1
    num_samples_per_prompt: int = 5
    seed: int = 41
    # only test the specified dimension, if empty list [], test all dims
    target_dimensions: List[str] = [
        # 'human_action',
        # 'scene',
        # 'multiple_objects',
        # 'appearance_style',
    ]


def perform_inference(pipe, data, args):
    
    prompt = data["prompt"]
    seed = data["seed"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_frames=args.video_frames
    resolution = args.res_dict[args.resolution]

    negative_prompt = """
    Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
    low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
    misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
    """
    
    start_time = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resolution['height'],
        width=resolution['width'],
        num_frames=num_frames,
        guidance_scale=args.cfg,
        num_inference_steps=args.diff_steps,
        generator=generator
    ).frames[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return {
        "output": output,
        "elapsed_time": elapsed_time,
    }


def load_vbench_prompts(prompt_json_path):
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data is a list，per-item like：
    # {
    #   "prompt_en": "...",
    #   "refined_prompt": "...",
    #   "dimension": ["subject_consistency", "aesthetic_quality", ...]
    # }
    return data


def main():
    args = InferArgs().parse_args()

    # *Initialize distributed process group
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    # tdist.init_process_group(backend='nccl')
    # explicitly set timeout to 3 hours
    tdist.init_process_group(
        backend='nccl', 
        timeout=datetime.timedelta(minutes=180)
    )
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()

    # ------ load HF WAN models ------
    print(f"[Rank {rank}] Loading models on device {device}...")

    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_type, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_type, subfolder="vae", torch_dtype=torch.float32)
    transformer = AutoModel.from_pretrained(
        args.model_type, subfolder="transformer", torch_dtype=torch.bfloat16)
    
    pipe = WanPipeline.from_pretrained(
        args.model_type,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    # 调整调度器：WAN官方推荐使用 UniPCMultistepScheduler
    scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=args.flow_shift,
    )
    pipe.scheduler = scheduler
    pipe.to(device)

    tdist.barrier(device_ids=[local_rank])      # wait all processes have loaded the model
    # --------------------------------

    # load vbench prompts
    all_vbench_items = load_vbench_prompts(args.prompt_json)
    
    # filter the specified dim
    if args.target_dimensions and len(args.target_dimensions) > 0:
        filtered_items = []
        target_set = set(args.target_dimensions)
        for item in all_vbench_items:
            dims = item.get("dimension", [])
            if any(d in target_set for d in dims):
                filtered_items.append(item)
        
        if rank == 0:
            print(f"\n[Info] Filtering prompts by dimensions: {args.target_dimensions}")
            print(f"[Info] Kept {len(filtered_items)} / {len(all_vbench_items)} prompts.")
        
        vbench_items = filtered_items
    else:
        vbench_items = all_vbench_items
    
    total_items = len(vbench_items)     # all dims are 946 prompts
    print(f"\n[Rank {rank}] Loaded {total_items} prompts from {args.prompt_json}")

    # *Distribute the data (prompts) across GPUs
    # per_gpu_prompts = (total_items + world_size - 1) // world_size
    # start_idx = rank * per_gpu_prompts
    # end_idx = min(start_idx + per_gpu_prompts, total_items)
    # print(f"[Rank {rank}] Processing prompts from index {start_idx} to {end_idx-1} (Total: {end_idx - start_idx})")

    videos_root = osp.join(args.output_root, "videos")
    videos_by_dim_root = osp.join(args.output_root, "videos_by_dimension")
    os.makedirs(videos_root, exist_ok=True)
    os.makedirs(videos_by_dim_root, exist_ok=True)
    if rank == 0:
        print(f"Saving raw videos to: {videos_root}")
        print(f"Saving dimension-organized videos to: {videos_by_dim_root}")
    
    local_total_latency = 0.0
    local_num_videos = 0
    local_num_skip_videos = 0       # for re-test after an interrupt
    local_tmp_fliker_dim_items = 0
    # warmup_steps = 2    # use 2 prompts for GPUs warm up
    # local_warmup_videos_count = warmup_steps * args.num_samples_per_prompt
    local_warmup_videos = 2
    
    local_indices = list(range(rank, total_items, world_size))
    print(f"[Rank {rank}] will process {len(local_indices)} prompts (Total tasks: {total_items})")
    print(f"[Rank {rank}] {local_indices=}")

    # for global_idx in trange(start_idx, end_idx, disable=(rank != 0), desc=f"Rank {rank} generating videos"):
    for global_idx in tqdm(local_indices, disable=(rank != 0), desc=f"Rank {rank} generating videos"):
        item = vbench_items[global_idx]
        dims = item.get("dimension", [])
        
        # Temporal Flickering dimension, sample 25 videos
        if "temporal_flickering" in dims:
            n_samples = 25
            local_tmp_fliker_dim_items += 1
        else:
            n_samples = args.num_samples_per_prompt
        
        # use refined_prompt, if None then use prompt_en
        # prompt = item.get("refined_prompt") or item.get("prompt_en")
        prompt = item.get("prompt_en")
        prompt_en = item.get("prompt_en")   # for video name like $prompt-$index.mp4
        prompt_seed = item.get("seed")
        if prompt is None:
            print(f"[Rank {rank}][Warning] No prompt found for index {global_idx}, skip.")
            continue
        # clean name
        # prompt_en = sanitize_filename(prompt_en)[:100] 
        
        # *Compute refined_prompt hash --> for prompt_en same but refined_prompt different !!!
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()[:6]

        for sample_idx in range(n_samples):
            # seed = args.seed + (global_idx * n_samples) + sample_idx
            seed = prompt_seed + n_samples * sample_idx
            print(f'\n[Rank {rank}] {global_idx=} {sample_idx=} {seed=} ==========')
            print(f'{prompt_en=} {dims=}')

            data = {
                'seed': seed,
                'prompt': prompt,
                'image_path': None,     # Vbench T2V
                'duration': args.generation_duration,
            }
            
            # base_name = f"idx{global_idx:04d}_s{sample_idx:02d}.mp4"
            base_name = f'{prompt_en}-{sample_idx}.mp4'
            physical_base_name = f'{prompt_en}-{prompt_hash}-{sample_idx}.mp4'
            save_path = osp.join(videos_root, physical_base_name)
            
            if osp.exists(save_path):
                print(f"[Rank {rank}] Skipping existing: {physical_base_name}")
                local_num_skip_videos += 1
                pass
            else:
                try:
                    output_dict = perform_inference(pipe, data, args)
                    video = output_dict["output"]
                    
                    local_num_videos += 1
                    if local_num_videos > local_warmup_videos:
                        local_total_latency += output_dict['elapsed_time']
                    
                    export_to_video(video, save_path, fps=args.fps)
                    # print(f"[Rank {rank}] Video genernation done: {save_path=}\n")
                
                except Exception as e:
                    print(f"[Rank {rank}] Error generating {physical_base_name}: {e}")
                    continue

            # a video prompt may belong multiple Vbench dims
            # create soft links under the dimension directory
            for d in dims:
                dim_dir = osp.join(videos_by_dim_root, d)
                os.makedirs(dim_dir, exist_ok=True)
                dim_video_path = osp.join(dim_dir, base_name)
                
                # [核心优化] 引入随机睡眠，错开 4 个进程的时间，极大降低碰撞概率
                # time.sleep(random.uniform(0.01, 0.05))
                
                if not osp.exists(dim_video_path):
                    try:
                        # print(f'[Rank {rank}] {save_path=} {dim_video_path=}')
                        os.symlink(osp.abspath(save_path), dim_video_path)
                        # print(f'[Rank {rank}] Create symlink ')
                    except FileExistsError:
                        print(f'[Rank {rank}] FileExistsError: {save_path=} {dim_video_path=}')
                        pass
                    except OSError:
                        # fall back to copy when no permission
                        import shutil
                        shutil.copy2(save_path, dim_video_path)
    
    print(f"[Rank {rank}] Finished tasks. Waiting for others...")
    tdist.barrier(device_ids=[local_rank])
    
    latency_tensor = torch.tensor(
        [local_total_latency, local_num_videos, local_num_skip_videos,
         local_tmp_fliker_dim_items],
        device=device,
        dtype=torch.float64
    )
    tdist.all_reduce(latency_tensor, op=tdist.ReduceOp.SUM)
    global_total_latency = latency_tensor[0].item()
    global_num_videos = int(latency_tensor[1].item())
    # Compute global warmup videos
    global_warmup_videos_count = local_warmup_videos * world_size
    # Profile global videos
    global_num_videos_profile = global_num_videos - global_warmup_videos_count
    global_num_skip_videos = int(latency_tensor[2].item())
    global_tmp_fliker_dim_items = int(latency_tensor[3].item())
    
    if rank == 0:
        avg_total_latency = global_total_latency / max(global_num_videos_profile, 1)
        throughput_total = global_num_videos_profile / global_total_latency if global_total_latency > 0 else 0
        total_videos = (total_items-global_tmp_fliker_dim_items)*args.num_samples_per_prompt + global_tmp_fliker_dim_items*25
        
        print("\n=== VBench Distributed Generation Benchmark Profile ===")
        print(f"Target Dimensions: {args.target_dimensions if args.target_dimensions else 'ALL'}")
        if global_num_skip_videos > 0:
            print(f"[Interrupt!] Skip videos: {total_videos-global_num_videos}({global_num_skip_videos})")
        print(f"Total videos: {global_num_videos_profile} + {global_warmup_videos_count}(warmup) = {global_num_videos}({total_videos})")
        print("---------- w/ decoder ----------")
        print(f"Total inference time: {global_total_latency:.2f} s")
        print(f"Average latency per video: {avg_total_latency:.4f} s")
        print(f"Throughput: {throughput_total:.4f} videos/sec")
        print("=======================================================\n")
    
        print("All videos generated.")


if __name__ == '__main__':
    main()

    # --- download HF weights ---
    # text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    # vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    # transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

    # text_encoder = UMT5EncoderModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16)
    # vae = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae", torch_dtype=torch.float32)
    # transformer = AutoModel.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="transformer", torch_dtype=torch.bfloat16)

