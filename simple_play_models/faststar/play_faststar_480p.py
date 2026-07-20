import os
import os.path as osp
import sys


REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from models.faststar.args_faststar import FastStarArgs
from simple_play_models.play_infinitystar_480p import perform_inference
from tools.run_infinity import InferencePipe, save_video
from utils.misc import time_str


class FastStar480pArgs(FastStarArgs):
    """FastSTAR defaults for 5-second, 81-frame 480p generation."""

    resolution: str = "480p"
    pn: str = "0.40M"
    fps: int = 16
    generation_duration: int = 5
    video_frames: int = 81

    model_path: str = "pretrained_models/infinitystar/infinitystar_8b_480p_weights"
    image_scale_repetition: str = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]"
    video_scale_repetition: str = "[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]"
    detail_scale_min_tokens: int = 350
    semantic_scales: int = 11

    # FastSTAR paper setting for the final four 480p scales.
    faststar_target_scales: str = "24,25,26,27"
    faststar_prune_ratios: str = "[0.20, 0.30, 0.40, 0.70]"
    faststar_mask_save_dir: str = "work_dir/play_models/FastSTAR_480p/faststar_masks"


if __name__ == "__main__":
    args = FastStar480pArgs().parse_args()
    pipe = InferencePipe(args)

    prompt = "A handsome smiling gardener inspecting plants, realistic cinematic lighting, detailed textures, ultra-realistic"
    # Set an image path for Image-to-Video generation.
    image_path = None
    data = {
        "seed": 41,
        "image_path": image_path,
        "prompt": prompt,
        "duration": args.generation_duration,
    }

    output_dict = perform_inference(pipe, data, args)
    save_dir = "work_dir/play_models/FastSTAR_480p"
    gen_video_path = osp.join(save_dir, "gen_videos", f"demo_{time_str()}.mp4")
    save_video(output_dict["output"], fps=args.fps, save_filepath=gen_video_path)

    print(
        f"FastSTAR 480p video generation done: {gen_video_path=}, "
        f"elapsed_time={output_dict['elapsed_time']:.3f}s"
    )
