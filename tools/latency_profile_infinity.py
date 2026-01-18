import random
from typing import Union
# import psutil
import pynvml
import torch
lib = torch.cuda.cudart()
import os
import os.path as osp
import cv2

from tqdm import tqdm

from models.scalekv.scale_kv import enable_scale_kv
from run_infinity import *
from tools.latency_profile import format_memory, get_memory_usage


# default_prompts = [
#     "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
#     "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur."
#     "Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.",
#     "A 4k dslr image of a lemur wearing a red magician hat and a blue coat performing magic tricks with cards in a garden.",
#     "A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid sytle of a vintage travel poster.",
#     "Crocodile in a sweater.",
#     "Luffy from ONEPIECE, handsome face, fantasy.",
#     "3d digital art of an adorable ghost, glowing within, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background.",
#     "an astronaut sitting in a diner, eating fries, cinematic, analog film",
#     "Chinese architecture, ancient style,mountain, bird, lotus, pond, big tree, 4K Unity, octane rendering.",
# ]
default_prompts = [
    'A high-contrast photo of a panda riding a horse. The panda is wearing a wizard hat and is reading a book. The horse is standing on a street against a gray concrete wall. Colorful flowers and the word "PEACE" are painted on the wall. Green grass grows from cracks in the street. DSLR photograph. daytime lighting.',
    'a red cube on the top of blue sphere, the behind is a yellow triangle. A cat is on the left and a dog is on the right',
    'The red hat was on the left of the blue backpack.',
    'woman lying on the beach',
    'woman laying on grass',
    'Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says "VAR" made out of colorful energy',
    'A board with text "Hello, VAR"',
    'A paper reads "No!"',
    'A photograph featuring a young woman standing in a field of tall green plants, possibly corn, with the sun shining through the foliage creating a warm, golden glow. The woman is looking off to the side with a gentle expression, and her face is partially obscured by the plants. The sunlight creates a lens flare effect, adding a dreamy quality to the image. The style of the image is naturalistic, capturing a moment of serenity in a rural setting.',
    "A photo-realistic picture. A black and white photograph that captures a man in profile. The man has a beard and mustache, and his hair appears to be swept back. He is wearing a scarf or shawl that is wrapped around his neck and shoulders, adding texture to the image. The photograph is taken from a close-up angle, focusing on the man's face and the upper part of his torso. The lighting is dramatic, with strong contrasts between light and shadow, highlighting the contours of his face and the texture of his hair and clothing. The style of the image is reminiscent of a cinematic or artistic portrait, emphasizing mood and emotion over realism.",
    'A man engaged in the activity of paddleboarding. He is balancing on a white paddleboard with a pink nose, which is partially submerged in the blue water. The man is wearing a black sleeveless top, blue shorts, and sunglasses. His hair is long and appears to be wet, suggesting he has been in the water. He is smiling and seems to be enjoying the moment, with his arms outstretched for balance. The background shows a clear sky and distant mountains, indicating that the setting is likely a large body of water, such as a lake or sea, on a sunny day. The photograph is taken in a realistic style, capturing the action and the natural environment.',
    'a young woman standing in the grass,, in the style of stark black-and-white photography,, hasselblad 1600f,, coastal landscapes,, expressive facial features,, dutch landscapes,, soft atmospheric scenes,, powerful portraits',
    'a digital painting of an old man with a beard and some dark grays,, in the style of photorealistic urban scenes,, uhd image,, algeapunk,, rusty debris,, vibrant portraits,, flickr,, soft-focus portraits',
    'beautiful female warrior,, short blue hair,, shimmering jewels armor,, in the style of Alfons Mucha,, with emphasis on light play and the transparency of the glass,, High and short depth of field,, Ray tracing,, FHD,, hyper quality',
    'a young female hobbit,, ultra realism,, lord of the rings,, snowy forest,, pale hues,, hobbit from lord of the rings who escaped Carn Dum,, grimy,, dirty,, black hair,, homely,, ugly',
    'A dog is walking on a leash with its owner.',
    'A man is running a marathon and crossing the finish line.',
    'an oblong eggplant and a teardrop pear',
    'an oblong cucumber and a teardrop pepper',
    'a brown dog and a blue horse',
    'a rabbit fights with a tiger',
    'three women',
    'three deer',
    'a tree',
    'a photo of a tree',
    'grassland',
    'a woman rides a tiger in the forest',
    'a beautiful scenery area of russia',
    'an oil painting of a house',
    "two girls",
    "three boys",
    'two candles on a marble table next to a silver spoon',
    'woman lying on the beach',
    'woman laying on grass',
    'woman laying on the beach',
    'liberty of statue',
    'a man full body shot',
    'a woman full body shot',
    'a set of sushi which consists of a US map shape',
    'Asian girl near the beach',
    'two women sitting in the sofa and hold red wine cup',
    'a rabbit fights with a tiger',
    'two ninjas fight with each other during night',
    'a red cube on the top of blue sphere, the behind is a yellow triangle. A cat is on the left and a dog is on the right',
    'Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says "VAR" made out of colorful energy',
    'a woman having a spa day',
    'two men boxing',
    'a Chinese woman laying on the beach',
    'a man laying on a bed',
    'A brand with "VAR DIT"',
    'A board with text "Hello, VAR"',
    'A paper reads "No!"',
    'A paper reads "VAR Yes!"',
    'American national flag',
    'China national flag',
    'Russia national flag',
    'a woman lying on the beach sunbathing',
    'ironman',
    "Generate the text 'happy' with autumn leaves and cold colors.",
    "Generate the text 'bytedance' with autumn leaves and cold colors.",
    "Generate the text 'GenAI' in a supermarket.",
    "Generate the text 'GenAI' in a grass.",
    "Generate the text 'GenAI' in a ground.",
    "Generate the text 'KCTG' in a book.",
    "Generate the text 'GenAI' in a table.",
    "a Chinese model is sitting on a train, magazine cover, photorealistic, futuristic style",
]


def encode_prompts(text_tokenizer, text_encoder, prompt: Union[str, List[str]], enable_positive_prompt=False):
    if isinstance(prompt, str):
        prompt = [prompt]
    
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    
    captions = prompt
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
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


def get_physical_device_index():
    """
    Gets the physical device index that pynvml can use,
    even when CUDA_VISIBLE_DEVICES is set.
    """
    # Get the logical device index from PyTorch
    logical_device_index = torch.cuda.current_device()

    # Get the CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

    if cuda_visible_devices:
        # If the variable is set, parse it and find the physical index
        try:
            # Split the comma-separated string into a list of device IDs
            visible_devices = [int(d.strip()) for d in cuda_visible_devices.split(',')]
            # Map the logical index to the physical index
            if logical_device_index < len(visible_devices):
                physical_device_index = visible_devices[logical_device_index]
                return physical_device_index
            else:
                # This case should ideally not happen if PyTorch is configured correctly
                raise IndexError("PyTorch logical device index is out of bounds of CUDA_VISIBLE_DEVICES.")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse CUDA_VISIBLE_DEVICES='{cuda_visible_devices}'. Error: {e}")
            # Fallback to using the logical index directly, which might be incorrect
            return logical_device_index
    else:
        # If the variable is not set, logical and physical indices are the same
        return logical_device_index


def main(args):
    # --- NVML init ---
    nvml_handle = None
    try:
        pynvml.nvmlInit()
        if torch.cuda.is_available():
            # Use the helper function to get the correct physical device index
            physical_device_index = get_physical_device_index()
            # nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            # -->
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_index)
            print(f"PyTorch is using device: cuda:{torch.cuda.current_device()}. Monitoring physical NVIDIA GPU index: {physical_device_index}")
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
        print("-------- Memory Usage (Initial) --------")
        print(f"CPU Memory: {format_memory(start_cpu)}")
        print(f"GPU Memory (PyTorch): {format_memory(start_pytorch_gpu)}")
        print(f"GPU Memory (NVIDIA-SMI): {format_memory(start_nvsmi_gpu)}")
        print("-" * 40)
        # --------------------------------------

        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)
        
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
        # seed = random.randint(0, 10000)
        seed = 42
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

        with torch.inference_mode():
            # for sparse attn layer count
            from models.sparvar.sparse_attn_layer_counter import singleton as layer_counter

            if not isinstance(cfg, list):
                cfg_list = [cfg] * len(scale_schedule)
            if not isinstance(tau, list):
                tau_list = [tau] * len(scale_schedule)

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
                text_cond_tuple = encode_prompts(text_tokenizer, text_encoder, prompts, enable_positive_prompt)
                if negative_prompt:
                    negative_label_B_or_BLT = encode_prompts(text_tokenizer, text_encoder, negative_prompt)
                else:
                    negative_label_B_or_BLT = None

                with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                    lib.cudaProfilerStart()
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
                    lib.cudaProfilerStop()
                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------
                layer_counter.reset()
            torch.cuda.synchronize(device=device)
            print("GPU warm-up finished.")
            
            # latency profile
            print(f"\nStarting inference speed test for {args.profile_iter} iterations...")
            timings = []
            sstart_time = time.time()
            for _ in tqdm(range(args.profile_iter)):
                prompts = random.sample(default_prompts, args.batch_size)
                print('\n====== prompts ======')
                print(f'{prompts}')
                print(f'=====================')

                text_cond_tuple = encode_prompts(text_tokenizer, text_encoder, prompts, enable_positive_prompt)
                if negative_prompt:
                    negative_label_B_or_BLT = encode_prompts(text_tokenizer, text_encoder, negative_prompt)
                else:
                    negative_label_B_or_BLT = None
                
                with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                    start_time = time.perf_counter()    # for accurate timing
                    lib.cudaProfilerStart()
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
                    lib.cudaProfilerStop()
                    
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time); print(f'%%%%%% {(end_time - start_time)/args.batch_size * 1000:.2f}ms %%%%%%')
                    
                    # --- Memory Analysis: Monitoring CPU and NVIDIA-SMI in a loop ---
                    cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
                    peak_cpu_mem = max(peak_cpu_mem, cpu)
                    peak_nvsmi_gpu_mem = max(peak_nvsmi_gpu_mem, nvsmi)
                    # ----------------------------------------------------------------
                layer_counter.reset()
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
    parser.add_argument('--outdir', type=str, default='')
    args = parser.parse_args()

    main(args)
