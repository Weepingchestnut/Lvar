import math
from typing import Dict, Iterable, List

from utils.arg_util_video import InferArgs


DEFAULT_FASTVAR_PRUNE_RATIOS = (0.20, 0.30, 0.40, 0.70)


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


class FastvarArgs(InferArgs):
    """Inference args for FastVAR-accelerated InfinityStar generation.

    FastVAR (ICCV'2025) is reproduced as the comparison baseline used in the FastSTAR
    paper: Pivotal Token Selection + Cached Token Restoration applied layer-wise
    (per Attention/FFN op) at the final high-resolution scales.
    """

    model_type: str = 'fastvar_infinitystar'

    fastvar_target_scales: str = ''                                             # default: last 4 scales
    fastvar_prune_ratios: str = '[0.20, 0.30, 0.40, 0.70]'                      # ratio == 1 skips the scale entirely (full pruning); Tap feeds these comments to argparse help, so never put a percent sign in them
    fastvar_cache_scale: int = -1                                               # -1: min(target_scales) - 1, i.e. the last un-pruned scale
    fastvar_prune_layer_range: str = '4,31'                                     # half-open [start, end); '' means all layers
    fastvar_final_iteration_full: int = 1                                       # choices=[0, 1]
    fastvar_per_frame_pts: int = 0                                              # choices=[0, 1]; 1: per-frame PTS mean & Top-K instead of clip-level
    fastvar_restore_interp_mode: str = 'area'                                   # choices=['area', 'trilinear', 'nearest']
    fastvar_log_pruning: int = 0

    def fastvar_target_scale_list(self, scale_schedule: Iterable) -> List[int]:
        """Return target scale indices, defaulting to the final 4 scales."""
        scale_schedule = list(scale_schedule)
        if len(scale_schedule) < 2:
            raise ValueError("FastVAR requires at least two scales (one cache scale + one pruned scale).")
        target_scales = _parse_int_list(self.fastvar_target_scales)
        if not target_scales:
            target_scales = list(range(max(len(scale_schedule) - 4, 1), len(scale_schedule)))   # 480p: [24, 25, 26, 27]
        if len(set(target_scales)) != len(target_scales):
            raise ValueError(f"FastVAR target scales must be unique, got {target_scales}.")
        invalid_scales = [scale for scale in target_scales if scale < 1 or scale >= len(scale_schedule)]
        if invalid_scales:
            raise ValueError(
                f"FastVAR target scales must be in [1, {len(scale_schedule) - 1}], got {invalid_scales}."
            )
        return sorted(target_scales)

    def fastvar_prune_ratio_list(self, target_scales: Iterable[int]) -> List[float]:
        """Return one prune ratio for each target scale."""
        target_scales = list(target_scales)
        prune_ratios = _parse_float_list(self.fastvar_prune_ratios)
        if not prune_ratios:
            prune_ratios = list(DEFAULT_FASTVAR_PRUNE_RATIOS)
        if len(prune_ratios) == 1 and len(target_scales) > 1:
            prune_ratios = prune_ratios * len(target_scales)
        if len(prune_ratios) != len(target_scales):
            raise ValueError(
                "FastVAR expects one prune ratio per target scale "
                f"({len(prune_ratios)} ratios for {len(target_scales)} scales)."
            )
        invalid_ratios = [ratio for ratio in prune_ratios if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0]
        if invalid_ratios:
            raise ValueError(
                f"FastVAR prune ratios must be finite values in [0, 1] "
                f"(ratio == 1 skips the scale entirely), got {invalid_ratios}."
            )
        return prune_ratios

    def fastvar_prune_ratio_by_scale(self, scale_schedule: Iterable) -> Dict[int, float]:
        """Return a scale-index to prune-ratio mapping for this run."""
        scale_schedule = list(scale_schedule)
        target_scales = self.fastvar_target_scale_list(scale_schedule)
        prune_ratios = self.fastvar_prune_ratio_list(target_scales)
        prune_ratio_by_scale = dict(zip(target_scales, prune_ratios))
        # ratio == 1 means the scale is skipped entirely; a skipped scale never writes its
        # KV cache, so later scales referencing it via context_info would crash. Hence
        # 100%-pruned scales must form the trailing suffix of the schedule.
        skip_scales = sorted(si for si, ratio in prune_ratio_by_scale.items() if ratio >= 1.0)
        if skip_scales:
            expected_suffix = list(range(len(scale_schedule) - len(skip_scales), len(scale_schedule)))
            if skip_scales != expected_suffix:
                raise ValueError(
                    "FastVAR 100% pruning (scale skipping) is only supported for the trailing "
                    f"scales of the schedule, got skip scales {skip_scales} "
                    f"for a {len(scale_schedule)}-scale schedule."
                )
        return prune_ratio_by_scale

    def fastvar_cache_scale_index(self, target_scales: Iterable[int]) -> int:
        """Return the cache scale (CTR restoration source), i.e. the last full-token scale."""
        target_scales = list(target_scales)
        cache_scale = int(self.fastvar_cache_scale)
        if cache_scale < 0:
            cache_scale = min(target_scales) - 1
        if cache_scale < 0 or cache_scale >= min(target_scales):
            raise ValueError(
                f"FastVAR cache scale must be a full-token scale before every target scale, "
                f"got cache_scale={cache_scale} for target scales {target_scales}."
            )
        return cache_scale

    def fastvar_prune_layer_list(self, depth: int) -> List[int]:
        """Return the transformer layer indices that apply token pruning ('' means all layers)."""
        value = str(self.fastvar_prune_layer_range).strip()
        if not value:
            return list(range(depth))
        bounds = _parse_int_list(value)
        if len(bounds) != 2:
            raise ValueError(f"FastVAR prune layer range expects 'start,end', got {value!r}.")
        start, end = bounds
        if not (0 <= start <= end <= depth):
            raise ValueError(f"FastVAR prune layer range must satisfy 0 <= start <= end <= {depth}, got {bounds}.")
        return list(range(start, end))

    def fastvar_should_prune(
            self,
            scale_index: int,
            repeat_index: int,
            infer_repeat_times: int,
            prune_ratio_by_scale: Dict[int, float],
    ) -> bool:
        if scale_index not in prune_ratio_by_scale:
            return False
        if (
            bool(int(self.fastvar_final_iteration_full))
            and infer_repeat_times > 1
            and repeat_index == infer_repeat_times - 1
        ):
            return False
        return True
