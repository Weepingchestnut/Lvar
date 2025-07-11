import argparse
import json
import torch.distributed as tdist

from lightning_fabric import seed_everything
from tqdm import trange

from models.scalekv.scale_kv import enable_scale_kv
from tools.conf import HF_HOME, HF_TOKEN
from tools.run_infinity import *


# set environment variables
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HF_HOME'] = HF_HOME
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--metadata_file', type=str, default='evaluation/gen_eval/prompts/evaluation_metadata.jsonl')
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--load_rewrite_prompt_cache', type=int, default=1, choices=[0,1])
    args = parser.parse_args()

    # parse cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    # *Initialize distributed process group
    tdist.init_process_group(backend='nccl')
    rank = tdist.get_rank()
    world_size = tdist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    with open(args.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    # *Distribute the data (prompts) across GPUs
    total_samples = len(metadatas)
    per_gpu = (total_samples + world_size - 1) // world_size
    start_idx = rank * per_gpu
    end_idx = min(start_idx + per_gpu, total_samples)
    
    prompt_rewrite_cache_file = osp.join('evaluation/gen_eval', 'prompt_rewrite_cache.json')
    if osp.exists(prompt_rewrite_cache_file):
        with open(prompt_rewrite_cache_file, 'r') as f:
            prompt_rewrite_cache = json.load(f)
            print(f"\n[Rank {rank}: Load prompt_rewrite_cache.json successful!]\n")
    else:
        prompt_rewrite_cache = {}
        print(f"\n[Rank {rank}: Load prompt_rewrite_cache.json false!]\n")
    
    if args.model_type == 'flux_1_dev':
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
    elif args.model_type == 'flux_1_dev_schnell':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    elif 'infinity' in args.model_type:
        print(f"[Rank {rank}] Loading Infinity model...")
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity = load_transformer(vae, args)

        if 'scalekv' in args.model_type:
            infinity = enable_scale_kv(infinity, window_size=16, max_capacity=650, kernel_size=5, pooling='maxpool')

        # if args.rewrite_prompt:
        #     from tools.prompt_rewriter import PromptRewriter
        #     prompt_rewriter = PromptRewriter(system='', few_shot_history=[])
    
    # for index, metadata in enumerate(metadatas):
    for index in trange(start_idx, end_idx, disable=rank!=0, desc=f"Rank {rank}"):
        seed_everything(args.seed)
        # seed_everything(args.seed + index) # Use a different seed for each prompt
        metadata = metadatas[index]

        outpath = os.path.join(args.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata['prompt']
        print(f"[Rank {rank}] Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        tau = args.tau
        cfg = args.cfg
        if args.rewrite_prompt:
            old_prompt = prompt
            if args.load_rewrite_prompt_cache and prompt in prompt_rewrite_cache:
                prompt = prompt_rewrite_cache[prompt]
            # else:
            #     refined_prompt = prompt_rewriter.rewrite(prompt)
            #     input_key_val = extract_key_val(refined_prompt)
            #     prompt = input_key_val['prompt']
            #     prompt_rewrite_cache[prompt] = prompt
            print(f'[Rank {rank}] old_prompt: {old_prompt}, refined_prompt: {prompt}')
            
        images = []
        for sample_j in range(args.n_samples):
            print(f"[Rank {rank}] Generating {sample_j+1} of {args.n_samples}, prompt={prompt}")
            
            t1 = time.time()
            if args.model_type == 'flux_1_dev':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                    num_images_per_prompt=1,
                ).images[0]
            elif args.model_type == 'flux_1_dev_schnell':
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=torch.Generator("cpu").manual_seed(0)
                ).images[0]
            elif args.model_type == 'pixart_sigma':
                image = pipe(prompt).images[0]
            elif 'infinity' in args.model_type:
                h_div_w_template = 1.000
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
                tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

                image = gen_one_img(
                    infinity, vae, text_tokenizer, text_encoder, 
                    prompt, tau_list=tau, cfg_sc=3, cfg_list=cfg, 
                    scale_schedule=scale_schedule, 
                    cfg_insertion_layer=[args.cfg_insertion_layer], 
                    vae_type=args.vae_type
                )
            else:
                raise ValueError
            t2 = time.time()
            print(f'[Rank {rank}] {args.model_type} infer one image takes {t2-t1:.2f}s')
            images.append(image)
        for i, image in enumerate(images):
            save_file = os.path.join(sample_path, f"{i:05}.jpg")
            if 'infinity' in args.model_type:
                cv2.imwrite(save_file, image.cpu().numpy())
            else:
                image.save(save_file)
    
    tdist.barrier()
    if rank == 0:
        print("All processes finished generation.")
        # with open(prompt_rewrite_cache_file, 'w') as f:
        #     json.dump(prompt_rewrite_cache, f, indent=2)
