import os
from PIL import Image
from models.grn.grn_pipeline import GRNPipeline
from utils.misc import time_str


def main():

    # prompt="A cute cat playing in the garden"
    prompt = "A close-up portrait in a cinematic photography style, capturing a girl-next-door look on a sunny daytime urban street. She wears a khaki sweater, with long, flowing hair gently draped over her shoulders. Her head is turned slightly, revealing soft facial features illuminated by realistic, delicate sunlight coming from the left. The sunlight subtly highlights individual strands of her hair. The image has a Canon film-like color tone, evoking a warm nostalgic atmosphere."
    # prompt = """a Chinese model is sitting on a train, magazine cover, clothes made of plastic, photorealistic, futuristic style, gray and green light, movie lighting, 32K HD"""

    # Load pipeline
    pipeline = GRNPipeline.from_pretrained(
        # hf_repo_id='bytedance-research/GRN',
        model_path='pretrained_models/grn/GRN_T2I_2B.pth',
        vae_path='pretrained_models/grn/HBQ_tokenizer_64dim_M4.ckpt',
        text_encoder_ckpt='pretrained_models/grn/umt5-xxl',
        # 
        task='T2I',
        pn='1M', 
        device='cpu',
    ).to('cuda')

    # Generate one image
    result = pipeline(
        prompt=prompt,
        guidance_scale=3.0,
        temperature=1.1,
        complexity_aware_Tmin=10,
        complexity_aware_Tmax=50,
        complexity_aware_k = 0,
        complexity_aware_b = 50,
        complexity_aware_wp = 5,
        snr_shift = 1.,
        h_div_w=1.,
        content_type='image',
        seed=42,
    )
    image = result.images[0]

    save_path = 'work_dir/play_models/GRN'
    img_path = os.path.join(save_path, f"example_{time_str()}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    image.save(img_path)
    # print(f'\nImage saved as {os.path.abspath(img_path)}')
    print(f'\nImage saved as {img_path}')


if __name__ == "__main__":
    main()
