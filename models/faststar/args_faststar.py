import math
from typing import Dict, Iterable, List

from utils.arg_util_video import InferArgs


DEFAULT_FASTSTAR_PRUNE_RATIOS = (0.20, 0.30, 0.40, 0.70)


def _parse_int_list(value) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _parse_float_list(value) -> List[float]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _parse_p_norm(value) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
        return math.inf
    return float(value)


class FastStarArgs(InferArgs):
    """Inference args for FastSTAR-enabled InfinityStar generation."""

    model_type: str = 'faststar_qwen8b'

    faststar_target_scales: str = ''                                            # default: last 4 scales
    faststar_prune_ratios: str = '[0.20, 0.30, 0.40, 0.70]'
    faststar_p_norm: str = '2'
    faststar_per_frame_topk: int = 1                                            # choices=[0, 1]
    faststar_first_clip_temporal_fallback: str = "spatial_only"
    faststar_final_iteration_full: int = 0                                      # choices=[0, 1]

    faststar_log_masks: int = 1
    faststar_save_masks: int = 1
    faststar_mask_save_dir: str = "work_dir/play_models/FastSTAR_720p/faststar_masks"

    def faststar_target_scale_list(self, scale_schedule: Iterable) -> List[int]:
        """Return target scale indices, defaulting to the final 4 scales."""
        scale_schedule = list(scale_schedule)
        target_scales = _parse_int_list(self.faststar_target_scales)
        if not target_scales:
            target_scales = list(range(max(len(scale_schedule) - 4, 0), len(scale_schedule)))   # 720p: [26, 27, 28, 29]
        return target_scales

    def faststar_prune_ratio_list(self, target_scales: Iterable[int]) -> List[float]:
        """Return one prune ratio for each target scale."""
        target_scales = list(target_scales)
        prune_ratios = _parse_float_list(self.faststar_prune_ratios)
        if not prune_ratios:
            prune_ratios = list(DEFAULT_FASTSTAR_PRUNE_RATIOS)
        if len(prune_ratios) == 1 and len(target_scales) > 1:
            prune_ratios = prune_ratios * len(target_scales)
        if len(prune_ratios) != len(target_scales):
            raise ValueError(
                "FastSTAR expects one prune ratio per target scale "
                f"({len(prune_ratios)} ratios for {len(target_scales)} scales)."
            )
        return prune_ratios

    def faststar_prune_ratio_by_scale(self, scale_schedule: Iterable) -> Dict[int, float]:
        """Return a scale-index to prune-ratio mapping for this run."""
        target_scales = self.faststar_target_scale_list(scale_schedule)
        prune_ratios = self.faststar_prune_ratio_list(target_scales)
        return dict(zip(target_scales, prune_ratios))

    def faststar_p_norm_value(self) -> float:
        return _parse_p_norm(self.faststar_p_norm)

    def faststar_should_prune(
            self,
            scale_index: int,
            repeat_index: int,
            infer_repeat_times: int,
            prune_ratio_by_scale: Dict[int, float],
    ) -> bool:
        if scale_index not in prune_ratio_by_scale:
            return False
        if (
            bool(int(self.faststar_final_iteration_full))
            and infer_repeat_times > 1
            and repeat_index == infer_repeat_times - 1
        ):
            return False
        return True
