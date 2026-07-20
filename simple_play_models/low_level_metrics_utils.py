"""Single-pair low-level metric helper for simple_play_models demo scripts.

Thin wrapper around evaluation/vbench/eval_low_level_metrics.py's single-pair
evaluation (PSNR / SSIM / LPIPS with the exact same metric configuration and
frame alignment rules) so a play script can compare its freshly generated demo
video against a saved baseline video without launching the full benchmark
pipeline. Heavy dependencies (torch, torchmetrics/lpips, video decoders) are
imported lazily and every failure degrades to a printed warning + None return,
so a long demo generation is never lost to a metric-side problem.

Note: a single demo pair is a quick sanity check, not a substitute for the
multi-video benchmark averages (FastSTAR computes these metrics over 10 videos
per VBench dimension). Metrics are computed on the encoded video files, the
same source the benchmark script uses for flat video directories.
"""

import json
import os
import os.path as osp
from datetime import datetime
from typing import Dict, Optional

_TAG = "[LowLevelMetrics]"


def evaluate_low_level_metrics_vs_baseline(
    candidate_video_path: str,
    baseline_video_path: str,
    device: str = "cuda",
    frame_batch_size: int = 16,
    lpips_net_type: str = "vgg",
    include_first_frame: bool = True,
    allow_frame_count_mismatch: bool = False,
    allow_spatial_mismatch: bool = False,
    save_json_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Compare one generated video against one baseline video.

    Returns {"psnr", "ssim", "lpips", "num_frames"} (frame-averaged, protocol
    identical to eval_low_level_metrics.py) or None when evaluation could not
    run. Mismatched frame counts / spatial sizes are treated as errors unless
    the corresponding allow_* flag is set, mirroring the benchmark defaults.
    """
    if not osp.isfile(baseline_video_path):
        print(f"{_TAG} baseline video not found, skip evaluation: {baseline_video_path}")
        return None
    if not osp.isfile(candidate_video_path):
        print(f"{_TAG} candidate video not found, skip evaluation: {candidate_video_path}")
        return None

    try:
        import torch
        from evaluation.vbench.eval_low_level_metrics import (
            VideoPair, build_metric_modules, choose_decoder_backend,
            evaluate_video_pair)
    except ImportError as exc:
        print(f"{_TAG} evaluation modules unavailable ({exc}); skip evaluation.")
        return None

    metric_device = torch.device(
        device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if metric_device.type == "cuda":
        # The generator model is usually still resident when a play script
        # calls this helper; release cached blocks before loading LPIPS.
        torch.cuda.empty_cache()

    try:
        metrics = build_metric_modules(device=metric_device, lpips_net_type=lpips_net_type)
        decoder_backend = choose_decoder_backend("auto")
    except ImportError as exc:
        print(f"{_TAG} metric dependencies unavailable ({exc}); skip evaluation.")
        return None

    pair = VideoPair(
        key=osp.basename(candidate_video_path),
        baseline_path=baseline_video_path,
        candidate_path=candidate_video_path,
    )
    try:
        result, error = evaluate_video_pair(
            pair=pair,
            decoder_backend=decoder_backend,
            device=metric_device,
            frame_batch_size=frame_batch_size,
            allow_frame_count_mismatch=allow_frame_count_mismatch,
            allow_spatial_mismatch=allow_spatial_mismatch,
            include_first_frame=include_first_frame,
            metrics=metrics,
        )
    except Exception as exc:  # decode/runtime failures should not kill the demo
        result, error = None, f"{type(exc).__name__}: {exc}"
    if error is not None:
        print(f"{_TAG} evaluation failed: {error}")
        return None

    assert result is not None
    print(f"{_TAG} candidate: {candidate_video_path}")
    print(f"{_TAG} baseline : {baseline_video_path}")
    print(
        f"{_TAG} frames={result['num_frames']}  "
        f"PSNR={result['psnr']:.2f} dB  SSIM={result['ssim']:.4f}  LPIPS={result['lpips']:.4f}"
    )

    if save_json_path:
        report = {
            "timestamp": datetime.now().isoformat(),
            "candidate_video": osp.abspath(candidate_video_path),
            "baseline_video": osp.abspath(baseline_video_path),
            "decoder_backend": decoder_backend,
            "device": str(metric_device),
            "frame_batch_size": frame_batch_size,
            "lpips_net_type": lpips_net_type,
            "include_first_frame": include_first_frame,
            "metrics": result,
        }
        save_json_path = osp.abspath(save_json_path)
        os.makedirs(osp.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"{_TAG} saved metric report to: {save_json_path}")

    return result
