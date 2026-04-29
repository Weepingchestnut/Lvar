import datetime
import hashlib
import json
import os
import os.path as osp
import random
import shutil
import sys
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.distributed as tdist
from PIL import Image
from tqdm import tqdm

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.infinitystar.self_correction import SelfCorrection
from models.schedules import get_encode_decode_func
from models.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index)
from tools.run_infinity import (InferencePipe, gen_one_video, 
                                save_video, transform)
from utils.arg_util_video import InferArgs


def perform_inference(pipe, data, args):
    
    prompt = data["prompt"]
    seed = data["seed"]
    mapped_duration=args.generation_duration    # default: 5 seconds
    num_frames=args.video_frames                # default: 81 frames

    # If an image_path is provided, perform image-to-video generation.
    image_path = data.get("image_path", None)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-0.571))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
    args.tower_split_index = args.first_full_spatial_size_scale_index + 1
    context_info = pipe.get_scale_pack_info(scale_schedule, args.first_full_spatial_size_scale_index, args)    
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['pt2scale_schedule'][(num_frames-1)//4+1]
    tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(scale_schedule) - args.tower_split_index)
    tgt_h, tgt_w = scale_schedule[-1][1] * 16, scale_schedule[-1][2] * 16
    gt_leak, gt_ls_Bl = -1, None

    if image_path is not None:
        ref_image = [cv2.imread(image_path)[:,:,::-1]]
        ref_img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in ref_image]
        ref_img_T3HW = torch.stack(ref_img_T3HW, 0) # [t,3,h,w]
        ref_img_bcthw = ref_img_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        _, _, gt_ls_Bl, _, _, _ = pipe.video_encode(pipe.vae, ref_img_bcthw.cuda(), vae_features=None, self_correction=pipe.self_correction, args=args, infer_mode=True, dynamic_resolution_h_w=dynamic_resolution_h_w)
        gt_leak=len(scale_schedule)//2

    generated_image_list = []
    negative_prompt=''
    if args.append_enlarge2captain:
        prompt = f'{prompt}, Close-up on big objects, emphasize scale and detail'       # 特写大型物体，突出尺寸与细节
    if args.append_duration2caption:
        prompt = f'<<<t={mapped_duration}s>>>' + prompt
    
    start_time = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True, cache_enabled=True), torch.no_grad():
        generated_image, _ = gen_one_video(
            pipe.infinity,
            pipe.vae,
            pipe.text_tokenizer,
            pipe.text_encoder,
            prompt,
            negative_prompt=negative_prompt,
            g_seed=seed,
            gt_leak=gt_leak,
            gt_ls_Bl=gt_ls_Bl,
            cfg_list=args.cfg, 
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[0],
            vae_type=args.vae_type,
            sampling_per_bits=1,
            enable_positive_prompt=0,
            low_vram_mode=True,
            args=args,
            get_visual_rope_embeds=pipe.get_visual_rope_embeds,
            context_info=context_info,
            noise_list=None,
        )
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        # print(generated_image.shape)
        generated_image_list.append(generated_image)
            
    generated_image = torch.cat(generated_image_list, 2)
    end_time = time.time()
    elapsed_time = end_time - start_time    
    
    return {
        "output": generated_image.cpu().numpy(),
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


def get_video_artifact_paths(video_save_path):
    video_stem, _ = osp.splitext(video_save_path)
    return {
        "video": video_save_path,
        "png_dir": video_stem,
        "npy": f"{video_stem}.npy",
    }


def has_complete_saved_outputs(video_save_path, args):
    artifact_paths = get_video_artifact_paths(video_save_path)
    if not osp.isfile(artifact_paths["video"]):
        return False

    if args.save_raw_npy_frames and not osp.isfile(artifact_paths["npy"]):
        return False

    if args.save_raw_png_frames:
        png_dir = artifact_paths["png_dir"]
        if not osp.isdir(png_dir):
            return False
        png_count = len([name for name in os.listdir(png_dir) if name.lower().endswith(".png")])
        expected_png_count = max(args.video_frames, 1)
        if png_count < expected_png_count:
            return False

    return True


def create_symlink_or_copy(src_path, dst_path):
    src_path = osp.abspath(src_path)
    dst_dir = osp.dirname(dst_path)
    os.makedirs(dst_dir, exist_ok=True)
    relative_src_path = osp.relpath(src_path, start=dst_dir)

    if osp.lexists(dst_path):
        if osp.islink(dst_path):
            current_link_target = os.readlink(dst_path)
            current_resolved_path = osp.abspath(osp.join(dst_dir, current_link_target))
            if current_link_target == relative_src_path and current_resolved_path == src_path and osp.exists(current_resolved_path):
                return
            os.unlink(dst_path)
        else:
            return
    try:
        os.symlink(relative_src_path, dst_path)
    except FileExistsError:
        print(f'FileExistsError: {src_path=} {dst_path=}')
    except OSError:
        if osp.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


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

    # load models
    print(f"[Rank {rank}] Loading models on device {device}...")
    pipe = InferencePipe(args, device)
    tdist.barrier(device_ids=[local_rank])      # wait all processes have loaded the model

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
        prompt = item.get("refined_prompt") or item.get("prompt_en")
        prompt_en = item.get("prompt_en")   # for video name like $prompt-$index.mp4
        prompt_seed = item.get("seed")
        if prompt is None:
            print(f"[Rank {rank}][Warning] No prompt found for index {global_idx}, skip.")
            continue
        # clean name
        # prompt_en = sanitize_filename(prompt_en)[:100] 
        
        # Compute refined_prompt hash --> for prompt_en same but refined_prompt different
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
            artifact_paths = get_video_artifact_paths(save_path)
            
            if has_complete_saved_outputs(save_path, args):
                print(f"[Rank {rank}] Skipping existing complete outputs: {physical_base_name}")
                local_num_skip_videos += 1
                pass
            else:
                try:
                    output_dict = perform_inference(pipe, data, args)
                    video_np = output_dict["output"]  # [bs, t, h, w, 3] in uint8
                    
                    local_num_videos += 1
                    if local_num_videos > local_warmup_videos:
                        local_total_latency += output_dict['elapsed_time']
                    
                    save_video(
                        video_np,
                        fps=args.fps,
                        save_filepath=save_path,
                        save_raw_png_frames=bool(args.save_raw_png_frames),
                        save_raw_npy_frames=bool(args.save_raw_npy_frames),
                    )
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
                dim_stem, _ = osp.splitext(dim_video_path)
                base_stem, _ = osp.splitext(base_name)
                
                # [核心优化] 引入随机睡眠，错开 4 个进程的时间，极大降低碰撞概率
                # time.sleep(random.uniform(0.01, 0.05))
                
                create_symlink_or_copy(save_path, dim_video_path)
                if args.save_raw_png_frames:
                    create_symlink_or_copy(artifact_paths["png_dir"], dim_stem)
                if args.save_raw_npy_frames:
                    create_symlink_or_copy(artifact_paths["npy"], osp.join(dim_dir, f'{base_stem}.npy'))
    
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
