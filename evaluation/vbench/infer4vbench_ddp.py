import datetime
import hashlib
import json
import os
import os.path as osp
import random
import sys
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.distributed as tdist
from PIL import Image
from tqdm import tqdm

from utils.load import load_video_visual_tokenizer

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.infinitystar.self_correction import SelfCorrection
from models.schedules import get_encode_decode_func
from models.schedules.dynamic_resolution import (
    get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index)
from tools.run_infinity import (gen_one_video, load_tokenizer,
                                load_video_transformer, save_video, transform)
from utils.arg_util_video import Args


def _init_prompt_rewriter():
    from tools.prompt_rewriter import OpenAIGPTModel
    """Initialize the OpenAI GPT model."""
    # Initialize the OpenAI GPT model
    model_name = 'gpt-4o-2024-08-06'
    ak = os.environ.get("OPEN_API_KEY", "")
    if len(ak) == 0:
        raise ValueError("Please provide your OpenAI API key in the OPEN_API_KEY environment variable.")
    model = OpenAIGPTModel(model_name, ak, if_global=True)
    system_prompt = (
        "You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description to make the video more realistic and beautiful. 0. Preserve ALL information, including style words and technical terms. 1. If the subject is related to person, you need to provide a detailed description focusing on basic visual characteristics of the person, such as appearance, clothing, expression, posture, etc. You need to make the person as beautiful and handsome as possible. When the subject is only one person or object, do not use they to describe him/her/it to avoid confusion with multiple subjects. 2. If the input does not include style, lighting, atmosphere, you can make reasonable associations. 3. We only generate a four-second video based on your descriptions. So do not generate descriptions that are too long, too complex or contain too many activities. 4. You can add some descriptions of camera movements with regards to the scenes and allow the scenes to have very natural and coherent movements. 6. If the input is in Chinese, translate the entire description to English. 7. Output ALL must be in English. 8. Here are some expanded descriptions that can serve as examples: 1. The video begins with a distant aerial view of a winding river cutting through a rocky landscape, with the sun casting a soft glow over the scene. As the camera moves closer, the river's flow becomes more visible, and the surrounding terrain appears more defined. The camera continues to approach, revealing a steep cliff with a person sitting on its edge. The person is positioned near the top of the cliff, overlooking the river below. The camera finally reaches a close-up view, showing the person sitting calmly on the cliff, with the river and landscape fully visible in the background. 2. In a laboratory setting, a machine with a metallic structure and a green platform is seen. A small, clear plastic bottle is positioned on the green platform. The machine has a control panel with red and green lights on the right side. A nozzle is positioned above the bottle, and it begins to dispense liquid into the bottle. The liquid is dispensed in small droplets, and the nozzle moves slightly between each droplet. The background includes other laboratory equipment and a mesh-like structure. 3. The video shows a panoramic view of a cityscape with a prominent building featuring a green dome and ornate architecture in the center. Surrounding the main building are several other structures, including a white building with balconies on the left and a taller building with multiple windows on the right. In the background, there are hills with scattered buildings and greenery. The camera remains stationary, capturing the scene from a fixed position, with no noticeable changes in the environment or the buildings throughout the frames. 4. In a dimly lit room with red and blue lighting, a person holds up a smartphone to record a video of a band performing. The band members are seated, with one holding a guitar and another playing a double bass. The smartphone screen shows the band members being recorded, with the camera capturing their movements and expressions. The background includes a lamp and some furniture, adding to the cozy atmosphere of the scene. 5. In a grassy area with scattered trees, a large tree stands prominently in the center. A lion is perched on a thick branch of this tree, looking out into the distance. The sky is overcast, adding a somber tone to the scene. 6. A man in a green sweater holding a paper turns around and speaks to a group of people seated in a theater. He then points at a man in a yellow sweater sitting in the front row. The man in the yellow sweater looks at the paper in his hand and begins to speak. The man in the green sweater lowers his head and then looks up at the man in the yellow sweater again. 7. An elderly man, wearing a beige sweater over a yellow shirt, is sitting in front of a laptop. He holds a pair of glasses in his right hand and appears to be deep in thought, resting his head on his hand. He then raises the glasses and rubs his eyes with his fingers, showing signs of fatigue. After rubbing his eyes, he places the glasses on his sweater and looks down at the laptop screen. 8. A woman and a child are sitting at a table, each holding a pencil and coloring on a piece of paper. The woman is coloring a green leafy plant, while the child is coloring a red and blue object. The table has several colored pencils, a container filled with more pencils, and a few small colorful blocks. The woman is wearing a striped shirt, and the child is focused on their drawing. 9. A person wearing teal running shoes and colorful socks is running on a wet, sandy surface. The camera captures the movement of their legs and feet as they lift off the ground and land back, creating a clear shadow on the wet sand. The shadow elongates and shifts with each step, indicating the person's motion. The background remains consistent with the wet, textured sand, and the focus is solely on the runner's feet and their shadow. 10. A man is running along the shoreline of a beach, with the ocean waves gently crashing onto the shore. The sun is setting in the background, casting a warm glow over the scene. The man is wearing a light-colored jacket and shorts, and his hair is blowing in the wind as he runs. The water splashes around his legs as he moves forward, and his reflection is visible on the wet sand. The waves create a dynamic and lively atmosphere as they roll in and out."
    )
    gpt_model = OpenAIGPTModel(model_name, ak, if_global=True)
    return gpt_model, system_prompt


class InferencePipe:
    def __init__(self, args, device):
        # load text encoder
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        self.vae = load_video_visual_tokenizer(args)
        self.vae = self.vae.float().to(device)
        # load infinity
        self.infinity = load_video_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        
        self._models = [self.text_tokenizer, self.text_encoder, self.vae, self.infinity, self.self_correction]

        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)

        if args.enable_rewriter:
            self.gpt_model, self.system_prompt = _init_prompt_rewriter()


def perform_inference(pipe, data, args):
    
    prompt = data["prompt"]
    seed = data["seed"]
    mapped_duration=5
    num_frames=81

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
        print(generated_image.shape)
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


class InferArgs(Args):
    
    model_type: str = 'infinitystar_qwen8b'
    pn: str = '0.90M'  # Pixel numbers, '0.90M' 720p, '0.40M' 480p
    fps: int = 16
    generation_duration: int = 5
    video_frames: int = generation_duration * fps + 1
    
    model_path: str = 'pretrained_models/infinitystar/infinitystar_8b_720p_weights'
    checkpoint_type: str = 'torch_shard'   # omnistore
    vae_path: str = 'pretrained_models/infinitystar/infinitystar_videovae.pth'
    text_encoder_ckpt: str = 'pretrained_models/infinitystar/text_encoder/flan-t5-xl-official/'
    videovae: int = 10
    text_channels: int = 2048
    
    dynamic_scale_schedule: str = 'infinity_elegant_clip20frames_v2'
    bf16: int = 1   # choices=[0,1]
    use_apg: int = 1    # choices=[0,1]
    use_cfg: int = 0
    cfg: float = 34
    tau_image: float = 1
    tau_video: float = 0.4
    apg_norm_threshold: float = 0.05
    image_scale_repetition: str = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    video_scale_repetition: str = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    # 使用 Tap 的 List 支持，这样终端可以直接输入 --image_scale_repetition 3 3 3 ...
    # image_scale_repetition: List[int] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    append_enlarge2captain: int = 1
    append_duration2caption: int = 1
    use_two_stage_lfq: int = 1
    detail_scale_min_tokens: int = 750
    semantic_scales: int = 12
    max_repeat_times: int = 10000
    
    # For optimal performance, enabling the prompt rewriter is recommended.
    # To utilize the GPT model, ensure the following environment variables are set:
    # export OPEN_API_KEY="YOUR_API_KEY"
    # export GLOBAL_AZURE_ENDPOINT="YOUR_ENDPOINT"
    # *--> use official rewrite VBench_rewrited_prompt.json
    enable_rewriter: int = 0
    
    # -------- VBench Setting --------
    prompt_json: str = 'evaluation/vbench/VBench_rewrited_prompt_fixed_seed.json'
    output_root: str = 'work_dir/evaluation/vbench/infinitystar_480p_81frames'
    start_index: int = 0
    end_index: int = -1
    num_samples_per_prompt: int = 5
    seed: int = 41
    # only test the specified dimension, if empty list [], test all
    target_dimensions: List[str] = [
        # 'human_action',
        # 'scene',
        # 'multiple_objects',
        # 'appearance_style',
    ]


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
            
            if osp.exists(save_path):
                print(f"[Rank {rank}] Skipping existing: {physical_base_name}")
                local_num_skip_videos += 1
                pass
            else:
                try:
                    output_dict = perform_inference(pipe, data, args)
                    video_np = output_dict["output"]  # [bs, t, h, w, 3] in uint8
                    
                    local_num_videos += 1
                    if local_num_videos > local_warmup_videos:
                        local_total_latency += output_dict['elapsed_time']
                    
                    save_video(video_np, fps=args.fps, save_filepath=save_path)
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
