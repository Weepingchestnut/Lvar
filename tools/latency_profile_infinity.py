import random
import psutil
import pynvml
import torch
import os
import os.path as osp
import cv2

from tqdm import tqdm

from models.scalekv.scale_kv import enable_scale_kv
from run_infinity import *
from tools.latency_profile import format_memory, get_memory_usage


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


def encode_prompts(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    # print(f'prompt={prompt}')
    # captions = [prompt]
    captions = prompt
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
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


def main(args):
    # --- NVML init ---
    nvml_handle = None
    try:
        pynvml.nvmlInit()
        if torch.cuda.is_available():
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    except pynvml.NVMLError as e:
        print(f"Warning: Could not initialize NVML. NVIDIA-SMI usage will not be reported. Error: {e}")
    # ----------------

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    try:
        device = torch.device("cuda")
    
        # --- Memory Analysis: Initial State ---
        torch.cuda.empty_cache()
        start_cpu, start_pytorch_gpu, start_nvsmi_gpu = get_memory_usage(device, nvml_handle)
        print("--- Memory Usage (Initial) ---")
        print(f"CPU Memory: {format_memory(start_cpu)}")
        print(f"GPU Memory (PyTorch): {format_memory(start_pytorch_gpu)}")
        print(f"GPU Memory (NVIDIA-SMI): {format_memory(start_nvsmi_gpu)}")
        print("-" * 30)
        # --------------------------------------

        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)
        if 'scalekv' in args.model_type:
            infinity = enable_scale_kv(infinity, window_size=16, max_capacity=650, kernel_size=5, pooling='maxpool')
        
        # --- Memory analysis: After model load ---
        torch.cuda.synchronize()
        post_cpu, post_pytorch_gpu, post_nvsmi_gpu = get_memory_usage(device, nvml_handle)
        print("\n--- Memory Usage (After Model Load) ---")
        print(f"CPU Memory: {format_memory(post_cpu)} (Used: {format_memory(post_cpu - start_cpu)})")
        print(f"GPU Memory (PyTorch): {format_memory(post_pytorch_gpu)} (Used: {format_memory(post_pytorch_gpu - start_pytorch_gpu)})")
        print(f"GPU Memory (NVIDIA-SMI): {format_memory(post_nvsmi_gpu)} (Used: {format_memory(post_nvsmi_gpu - start_nvsmi_gpu)})")
        print("-" * 35)
        # ----------------------------------------

        # hyperparameter setting
        tau = args.tau; cfg = args.cfg
        h_div_w = 1/1 # Aspect Ratio
        seed = random.randint(0, 10000)
        enable_positive_prompt = 0

        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        negative_prompt=''
        cfg_sc=3
        top_k=900
        top_p=0.97
        cfg_exp_k=0.0
        gumbel=0
        softmax_merge_topk=-1
        gt_leak=0
        gt_ls_Bl=None

        prompts = random.sample(default_prompts, args.batch_size)

        with torch.inference_mode():
            if not isinstance(cfg, list):
                cfg_list = [cfg] * len(scale_schedule)
            if not isinstance(tau, list):
                tau_list = [tau] * len(scale_schedule)
            text_cond_tuple = encode_prompts(text_tokenizer, text_encoder, prompts, enable_positive_prompt)
            if negative_prompt:
                negative_label_B_or_BLT = encode_prompts(text_tokenizer, text_encoder, negative_prompt)
            else:
                negative_label_B_or_BLT = None
            
            # --- Memory Analysis: Preparing for Inference ---
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            peak_cpu_mem = post_cpu
            peak_nvsmi_gpu_mem = post_nvsmi_gpu
            # ------------------------------------------------
            
            # warmup
            print(f"\nStarting GPU warm-up for {args.warmup_iter} iterations...")
            with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                for _ in tqdm(range(args.warmup_iter)):
                    _, _, img_list = infinity.autoregressive_infer_cfg(
                        vae=vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=seed,
                        B=args.batch_size, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
                        cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
                        returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
                        cfg_exp_k=cfg_exp_k, cfg_insertion_layer=[args.cfg_insertion_layer],
                        vae_type=args.vae_type, softmax_merge_topk=softmax_merge_topk,
                        ret_img=True, trunk_scale=1000,
                        gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
                        sampling_per_bits=args.sampling_per_bits,
                    )
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

                text_cond_tuple = encode_prompts(text_tokenizer, text_encoder, prompts, enable_positive_prompt)
                if negative_prompt:
                    negative_label_B_or_BLT = encode_prompts(text_tokenizer, text_encoder, negative_prompt)
                else:
                    negative_label_B_or_BLT = None
                
                with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                    start_time = time.perf_counter()    # for accurate timing
                    _, _, img_list = infinity.autoregressive_infer_cfg(
                        vae=vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=seed,
                        B=args.batch_size, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
                        cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
                        returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
                        cfg_exp_k=cfg_exp_k, cfg_insertion_layer=[args.cfg_insertion_layer],
                        vae_type=args.vae_type, softmax_merge_topk=softmax_merge_topk,
                        ret_img=True, trunk_scale=1000,
                        gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
                        sampling_per_bits=args.sampling_per_bits,
                    )
                    torch.cuda.synchronize(device=device)   # *Important*: Ensure that all CUDA operations are completed before recording the time
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)
                    
                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------
            
            ttotal_time = time.time() - sstart_time
            print("Inference speed test finished.")
        
        average_time = ttotal_time / args.profile_iter
        print(f"\nGeneration with batch_size = {args.batch_size} take {average_time:2f}s per step (w/ text encoder).")

        batch_size = args.batch_size
        avg_latency = sum(timings) / len(timings)
        std_latency = torch.tensor(timings).std().item()
        throughput = batch_size / avg_latency if avg_latency > 0 else float('inf')

        print(f"\n--- Inference Performance ---")
        print(f"Batch Size: {batch_size}")
        print(f"Average Latency: {avg_latency * 1000:.2f} ms")
        print(f"Latency StdDev: {std_latency * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")
        print("-" * 30)

        # --- Memory Analysis: final report ---
        peak_pytorch_gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print("\n--- Memory Usage (Peak during Inference) ---")
        print(f"Peak CPU Memory: {format_memory(peak_cpu_mem)} (Total Used: {format_memory(peak_cpu_mem - start_cpu)})")
        print(f"Peak GPU Memory (PyTorch): {format_memory(peak_pytorch_gpu_mem)} (Total Used: {format_memory(peak_pytorch_gpu_mem - start_pytorch_gpu)})")
        print(f"Peak GPU Memory (NVIDIA-SMI): {format_memory(peak_nvsmi_gpu_mem)} (Total Used: {format_memory(peak_nvsmi_gpu_mem - start_nvsmi_gpu)})")
        print("-" * 45)

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
    parser.add_argument('--outdir', type=str, default='')
    # --- exp params --
    # parser.add_argument('--freeze_kv_cache_last_n_scales', type=int, default=4)
    parser.add_argument('--base_cache_scales', type=int, default=5)
    args = parser.parse_args()

    main(args)
