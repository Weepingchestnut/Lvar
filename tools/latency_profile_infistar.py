import os
import os.path as osp
import random
import sys
import time

import cv2
import numpy as np
import pynvml
import torch
from PIL import Image
from tqdm import tqdm

# Make sure repo root is importable no matter where the script is launched from.
sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from run_infinity import (InferencePipe, encode_video_prompt, save_video,
                          transform)

from models.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index)
from utils.arg_util_video import InferArgs
from utils.profile_utils import (default_video_prompts, format_memory,
                                 get_memory_usage, get_physical_device_index)


def prepare4infer(pipe, data, args):

    if not isinstance(data["prompt"], str):
        prompt = data["prompt"][0]      # only support batch size = 1
    else:
        prompt = data["prompt"]
    prompt_prefix = prompt[:30].replace(" ", "_")

    # seed = data["seed"]                         # Latency profile, fix seed = args.seed
    mapped_duration=args.generation_duration    # default: 5 seconds
    num_frames=args.video_frames                # default: 81 frames

    # If an image_path is provided, perform image-to-video generation.
    image_path = data.get("image_path", None)

    dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, num_frames)
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
    
    negative_prompt=''      # default: ''
    if args.append_enlarge2captain:
        prompt = f'{prompt}, Close-up on big objects, emphasize scale and detail'       # 特写大型物体，突出尺寸与细节
    if args.append_duration2caption:
        prompt = f'<<<t={mapped_duration}s>>>' + prompt

    return scale_schedule, gt_leak, gt_ls_Bl, context_info, tau, negative_prompt, prompt, prompt_prefix


def main():
    args = InferArgs().parse_args()
    
    if args.infer_batch_size != 1:
        raise ValueError("InfinityStar benchmark currently supports only --batch_size=1.")
    if args.use_apg + args.use_cfg != 1:
        raise ValueError("Exactly one of --use_apg or --use_cfg must be 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency profiling.")
    
    random.seed(args.seed)
    # path to save generated videos
    videos_root = osp.join(args.profile_output_root, "videos")
    os.makedirs(videos_root, exist_ok=True)

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

        # load infinitystar models
        pipe = InferencePipe(args, device)

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

            ###################
            #* warmup
            ###################
            print(f"\nStarting GPU warm-up for {args.warmup_iter} iterations...")
            for _ in tqdm(range(args.warmup_iter)):
                prompts = random.sample(default_video_prompts, args.infer_batch_size)

                data = {
                    'seed': args.seed,
                    'prompt': prompts,
                    'image_path': None,     # Vbench T2V
                    'duration': args.generation_duration,
                }
                scale_schedule, gt_leak, gt_ls_Bl, context_info, \
                    tau_list, negative_prompt, prompt, prompt_prefix = prepare4infer(pipe, data, args)
                cfg_list=args.cfg

                # --- text encode ---
                if not isinstance(cfg_list, list):
                    cfg_list = [cfg_list] * len(scale_schedule)
                if not isinstance(tau_list, list):
                    tau_list = [tau_list] * len(scale_schedule)
                text_cond_tuple = encode_video_prompt(
                    args.text_encoder_ckpt, pipe.text_tokenizer, pipe.text_encoder, prompt, 
                    enable_positive_prompt=0, low_vram_mode=True)
                if negative_prompt:
                    negative_label_B_or_BLT = encode_video_prompt(
                        args.text_encoder_ckpt, pipe.text_tokenizer, pipe.text_encoder, 
                        negative_prompt, low_vram_mode=True)
                else:
                    negative_label_B_or_BLT = None
                # print(f'cfg: {cfg_list}, tau: {tau_list}')

                with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                    lib.cudaProfilerStart()
                    _ = pipe.infinity.autoregressive_infer(
                        vae=pipe.vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=args.seed,
                        B=args.infer_batch_size, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
                        cfg_sc=3, cfg_list=cfg_list, tau_list=tau_list, top_k=900, top_p=0.97,
                        returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                        cfg_exp_k=0.0, cfg_insertion_layer=[0],
                        vae_type=args.vae_type, softmax_merge_topk=-1,
                        ret_img=True, trunk_scale=1000,
                        gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
                        sampling_per_bits=1,
                        input_use_interplote_up=False,
                        low_vram_mode=True,
                        args=args,
                        get_visual_rope_embeds=pipe.get_visual_rope_embeds,
                        context_info=context_info,
                        noise_list=None,
                        return_summed_code_only=False,
                        mode='',
                        former_clip_features=None,
                        first_frame_features=None,
                    )
                    lib.cudaProfilerStop()
                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------
            torch.cuda.synchronize(device=device)
            print("GPU warm-up finished.")

            ###################
            #* latency profile
            ###################
            print(f"\nStarting inference speed test for {args.profile_iter} iterations...")
            timings = []
            # timings_w_txt_encode = []
            sstart_time = time.time()
            for _ in tqdm(range(args.profile_iter)):
                prompts = random.sample(default_video_prompts, args.infer_batch_size)   # list['prompt1', 'prompt2']
                default_video_prompts.remove(prompts[0])

                data = {
                    'seed': args.seed,
                    'prompt': prompts,
                    'image_path': None,     # Vbench T2V
                    'duration': args.generation_duration,
                }
                scale_schedule, gt_leak, gt_ls_Bl, context_info, \
                    tau_list, negative_prompt, prompt, prompt_prefix = prepare4infer(pipe, data, args)
                cfg_list=args.cfg

                # start_time_w_txt_encode = time.perf_counter()
                # --- text encode ---
                if not isinstance(cfg_list, list):
                    cfg_list = [cfg_list] * len(scale_schedule)
                if not isinstance(tau_list, list):
                    tau_list = [tau_list] * len(scale_schedule)
                text_cond_tuple = encode_video_prompt(
                    args.text_encoder_ckpt, pipe.text_tokenizer, pipe.text_encoder, prompt, 
                    enable_positive_prompt=0, low_vram_mode=True)
                if negative_prompt:
                    negative_label_B_or_BLT = encode_video_prompt(
                        args.text_encoder_ckpt, pipe.text_tokenizer, pipe.text_encoder, 
                        negative_prompt, low_vram_mode=True)
                else:
                    negative_label_B_or_BLT = None
                # print(f'cfg: {cfg_list}, tau: {tau_list}')

                with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                    start_time = time.perf_counter()
                    lib.cudaProfilerStart()
                    out = pipe.infinity.autoregressive_infer(
                        vae=pipe.vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=args.seed,
                        B=args.infer_batch_size, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
                        cfg_sc=3, cfg_list=cfg_list, tau_list=tau_list, top_k=900, top_p=0.97,
                        returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                        cfg_exp_k=0.0, cfg_insertion_layer=[0],
                        vae_type=args.vae_type, softmax_merge_topk=-1,
                        ret_img=True, trunk_scale=1000,
                        gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
                        sampling_per_bits=1,
                        input_use_interplote_up=False,
                        low_vram_mode=True,
                        args=args,
                        get_visual_rope_embeds=pipe.get_visual_rope_embeds,
                        context_info=context_info,
                        noise_list=None,
                        return_summed_code_only=False,
                        mode='',
                        former_clip_features=None,
                        first_frame_features=None,
                    )   # tuple([], tensor[bs(1), frames(81), H, W, 3])
                    torch.cuda.synchronize(device=device)   # *Important*: Ensure that all CUDA operations are completed before recording the time
                    lib.cudaProfilerStop()

                    end_time = time.perf_counter()
                    timings.append(end_time - start_time); print(f'%%%%%% {(end_time - start_time)/args.infer_batch_size:.2f}s %%%%%%')
                    # timings_w_txt_encode.append(end_time - start_time_w_txt_encode)

                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------

                    # save video
                    _, img_list = out; img = img_list[0]    # [frames(81), H, W, 3]
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)
                
                video_np = img.cpu().numpy()      # [t, h, w, 3] uint8
                video_save_path = osp.join(videos_root, f'{prompt_prefix}.mp4')
                save_video(video_np, fps=args.fps, save_filepath=video_save_path)
                # print(f'Video generation done: {video_save_path}')

            ttotal_time = time.time() - sstart_time
            print("Inference speed test finished.")

        average_time = ttotal_time / args.profile_iter
        print(f"\nGeneration with batch_size = {args.infer_batch_size} take {average_time:2f}s per step (w/ text encoder).")

        batch_size = args.infer_batch_size
        avg_batch_latency = sum(timings) / len(timings)
        std_batch_latency = torch.tensor(timings).std().item()
        throughput = batch_size / avg_batch_latency if avg_batch_latency > 0 else float('inf')
        # per-image latency
        avg_latency = avg_batch_latency / batch_size
        std_latency = std_batch_latency / batch_size

        print(f"\n--- Inference Performance ---")
        print(f"Batch Size: {batch_size}")
        print(f"Average Latency: {avg_latency:.2f} s")
        print(f"Latency StdDev: {std_latency * 1000:.2f} ms")
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
            pynvml.nvmlShutdown()   # NVML clean


if __name__ == "__main__":
    main()

    # prompt sample test
    # random.seed(41)
    # len(default_video_prompts)
    # for _ in range(30):
    #     prompts = random.sample(default_video_prompts, 1)
    #     # default_video_prompts.remove(prompts[0])
    #     print(prompts[0][:30])

