import os
import os.path as osp
import sys


REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from models.sparsevar.args_sparsevar import SparsevarArgs
from simple_play_models.low_level_metrics_utils import \
    evaluate_low_level_metrics_vs_baseline
from simple_play_models.play_infinitystar_480p import perform_inference
from tools.run_infinity import InferencePipe, save_video
from utils.misc import time_str


class Sparsevar480pArgs(SparsevarArgs):
    """SparseVAR-InfinityStar defaults for 5-second, 81-frame 480p generation."""

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

    # FastSTAR baseline protocol: SparseVAR applied to the final four 480p scales;
    # SparseVAR hyperparameters (0.6 / 4 / 3 / 0.8) keep the SparsevarArgs defaults.
    sparsevar_target_scales: str = "24,25,26,27"
    # FastSTAR fixed-ratio protocol: pruning ratios of scales (24, 25, 26, 27)
    # = (20%, 30%, 40%, 70%) per the FastSTAR Table 3 footnote (its 720p
    # baseline tables align SparseVAR the same way). Set to "" to fall back to
    # SparseVAR-native dynamic-threshold selection.
    sparsevar_pruning_ratios: str = "0.2,0.3,0.4,0.7"

    # Match the InfinityStar 480p single-A100-40GB path.
    drop_uncond_last_scales: int = 0


if __name__ == "__main__":
    args = Sparsevar480pArgs().parse_args()
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
    save_dir = "work_dir/play_models/SparseVAR/InfinityStar_480p"
    gen_video_path = osp.join(save_dir, "gen_videos", f"demo_{time_str()}.mp4")
    save_video(output_dict["output"], fps=args.fps, save_filepath=gen_video_path)

    print(
        f"SparseVAR-InfinityStar 480p video generation done: {gen_video_path=}, "
        f"elapsed_time={output_dict['elapsed_time']:.3f}s"
    )

    if args.baseline_video_path:
        evaluate_low_level_metrics_vs_baseline(
            candidate_video_path=gen_video_path,
            baseline_video_path=args.baseline_video_path,
            save_json_path=osp.splitext(gen_video_path)[0] + "_low_level_metrics.json",
        )
