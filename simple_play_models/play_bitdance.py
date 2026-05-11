import os

from models.bitdance.t2i_pipeline import BitDanceT2IPipeline
from utils.misc import time_str


def main():
    model_path = 'pretrained_models/bitdance/14b_64x'
    # model_path = 'models/BitDance-14B-16x'
    device = 'cuda'

    pipe = BitDanceT2IPipeline(model_path=model_path, device=device)

    # prompt = "A close-up portrait in a cinematic photography style, capturing a girl-next-door look on a sunny daytime urban street. She wears a khaki sweater, with long, flowing hair gently draped over her shoulders. Her head is turned slightly, revealing soft facial features illuminated by realistic, delicate sunlight coming from the left. The sunlight subtly highlights individual strands of her hair. The image has a Canon film-like color tone, evoking a warm nostalgic atmosphere."
    prompt = """a Chinese model is sitting on a train, magazine cover, clothes made of plastic, photorealistic, futuristic style, gray and green light, movie lighting, 32K HD"""

    image = pipe.generate(
        prompt=prompt,
        height=1024,
        width=1024,
        num_sampling_steps=50, # adjust to 25 steps for faster inference, but may slightly reduce quality
        guidance_scale=7.5,
        num_images=1,
        seed=42
    )[0]

    save_path = 'work_dir/play_models/bitdance'
    img_path = os.path.join(save_path, f"example_{time_str()}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    image.save(img_path)
    print(f'\nImage saved as {os.path.abspath(img_path)}')


if __name__ == "__main__":
    main()
