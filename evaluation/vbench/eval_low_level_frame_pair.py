import argparse
import os.path as osp

import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR, SSIM, and LPIPS between two image frames."
    )
    parser.add_argument("--gt-frame", type=str, required=True, help="Path to the GT/baseline frame image.")
    parser.add_argument("--pred-frame", type=str, required=True, help="Path to the compared/candidate frame image.")
    parser.add_argument("--device", type=str, default="cuda", help="Metric device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--lpips-net-type",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
        help="Backbone used by torchmetrics LPIPS. Keep this aligned with eval_low_level_metrics.py.",
    )
    parser.add_argument(
        "--allow-spatial-mismatch",
        action="store_true",
        help="If set, resize pred frame to GT spatial size before metric computation.",
    )
    return parser.parse_args()


def load_frame_rgb(frame_path: str) -> torch.Tensor:
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255.0)


def build_metric_modules(device: torch.device, lpips_net_type: str):
    try:
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    except ImportError:
        try:
            from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        except ImportError as exc:
            raise ImportError(
                "torchmetrics is required. Please install torchmetrics and lpips in the evaluation environment."
            ) from exc
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise ImportError(
            "torchmetrics is required. Please install torchmetrics and lpips in the evaluation environment."
        ) from exc

    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0,
        reduction="sum",
        dim=(1, 2, 3),
    ).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        reduction="sum",
    ).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type=lpips_net_type,
        reduction="sum",
        normalize=True,
    ).to(device)
    lpips_metric.eval()
    return psnr_metric, ssim_metric, lpips_metric


def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    gt_frame = load_frame_rgb(args.gt_frame).to(device)
    pred_frame = load_frame_rgb(args.pred_frame).to(device)

    if tuple(gt_frame.shape[-2:]) != tuple(pred_frame.shape[-2:]):
        if not args.allow_spatial_mismatch:
            raise RuntimeError(
                f"Spatial size mismatch: gt={tuple(gt_frame.shape[-2:])}, "
                f"pred={tuple(pred_frame.shape[-2:])}. Use --allow-spatial-mismatch to resize pred."
            )
        pred_frame = torch.nn.functional.interpolate(
            pred_frame,
            size=tuple(gt_frame.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )

    psnr_metric, ssim_metric, lpips_metric = build_metric_modules(device, args.lpips_net_type)

    with torch.inference_mode():
        psnr = float(psnr_metric(pred_frame, gt_frame).item())
        psnr_metric.reset()
        ssim = float(ssim_metric(pred_frame, gt_frame).item())
        ssim_metric.reset()
        lpips = float(lpips_metric(pred_frame, gt_frame).item())
        lpips_metric.reset()

    print("========== Low-Level Frame Pair Metrics ==========")
    print(f"GT frame   : {osp.abspath(args.gt_frame)}")
    print(f"Pred frame : {osp.abspath(args.pred_frame)}")
    print(f"Device     : {device}")
    print(f"LPIPS net  : {args.lpips_net_type}")
    print(f"Shape      : {tuple(gt_frame.shape)}")
    print("--------------------------------------------------")
    print(f"PSNR       : {psnr:.6f}")
    print(f"SSIM       : {ssim:.6f}")
    print(f"LPIPS      : {lpips:.6f}")
    print("==================================================")


if __name__ == "__main__":
    main()
