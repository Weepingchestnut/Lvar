import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from models.faststar.args_faststar import FastStarArgs
from simple_play_models.play_infinitystar_720p import perform_inference
from tools.run_infinity import InferencePipe, save_video
from utils.misc import time_str


if __name__ == '__main__':
    args = FastStarArgs().parse_args()

    pipe = InferencePipe(args)

    prompt = 'A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic'
    # image_path = 'assets/reference_image.webp'  # Remove this for Text-to-Video (T2V) generation
    image_path = None
    data = {
        'seed': 41,
        'image_path': image_path,
        'prompt': prompt,
    }

    output_dict = perform_inference(pipe, data, args)
    save_dir = 'work_dir/play_models/FastSTAR_720p'
    gen_video_path = osp.join(save_dir, 'gen_videos', f'demo_{time_str()}.mp4')
    save_video(output_dict['output'], fps=args.fps, save_filepath=gen_video_path)

    print(f'FastSTAR video generation done: {gen_video_path=}')
