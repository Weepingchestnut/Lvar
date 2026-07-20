from typing import Dict, Iterable, List

from utils.arg_util_video import InferArgs


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


class SparsevarArgs(InferArgs):
    """Inference args for SparseVAR-accelerated InfinityStar generation.

    Default hyperparameters follow the image-Infinity SparseVAR setup
    (utils/arg_util.py sparsevar_* defaults), per the FastSTAR baseline
    protocol of applying SparseVAR as-is across the spatiotemporal pyramid.
    """

    model_type: str = 'sparsevar_infinitystar'

    sparsevar_target_scales: str = ''           # default: last 4 scales
    sparsevar_compress_ratio: float = 0.6       # threshold for the dynamic keep ratio, {0.5, 0.6, 0.7}
    sparsevar_local_window_size: int = 4        # anchor grid stride (alpha)
    sparsevar_specific_mse_layer: int = 4       # block-chunk index used for the MSE change map
    sparsevar_beta: float = 0.9                 # anchor cosine-similarity threshold
    sparsevar_final_iteration_full: int = 1     # choices=[0, 1]; 0 = prune every repeat of target scales
    # 1 = paper Algorithm 1 line 8 (anchor lattice always computed, carved
    # from the keep budget); 0 = released image code's effective B=1 behavior
    # (anchor merge never triggers, anchor copies may read uncomputed inputs).
    sparsevar_force_keep_anchors: int = 1       # choices=[0, 1]
    # Per-scale nominal pruning ratios aligned with the sorted target scales,
    # e.g. "0.2,0.3,0.4,0.7" (FastSTAR fixed-ratio comparison protocol);
    # '' = SparseVAR-native dynamic ratio from the compress_ratio threshold.
    sparsevar_pruning_ratios: str = '0.2,0.3,0.4,0.7'

    sparsevar_log_tokens: int = 0

    def sparsevar_target_scale_list(self, scale_schedule: Iterable) -> List[int]:
        """Return target scale indices, defaulting to the final 4 scales."""
        scale_schedule = list(scale_schedule)
        if len(scale_schedule) < 2:
            raise ValueError("SparseVAR requires at least two scales to build a cross-scale token plan.")
        target_scales = _parse_int_list(self.sparsevar_target_scales)
        if not target_scales:
            target_scales = list(range(max(len(scale_schedule) - 4, 1), len(scale_schedule)))   # 480p: [24, 25, 26, 27]
        if len(set(target_scales)) != len(target_scales):
            raise ValueError(f"SparseVAR target scales must be unique, got {target_scales}.")
        invalid_scales = [scale for scale in target_scales if scale < 1 or scale >= len(scale_schedule)]
        if invalid_scales:
            raise ValueError(
                f"SparseVAR target scales must be in [1, {len(scale_schedule) - 1}], got {invalid_scales}."
            )
        return target_scales

    def sparsevar_nominal_keep_ratios(self, target_scales: Iterable[int]) -> Dict[int, float]:
        """Map target scale -> nominal keep ratio (1 - pruning ratio); {} = dynamic mode."""
        ratios = _parse_float_list(self.sparsevar_pruning_ratios)
        if not ratios:
            return {}
        target_scales = sorted(target_scales)
        if len(ratios) != len(target_scales):
            raise ValueError(
                f"sparsevar_pruning_ratios needs one ratio per target scale "
                f"({len(target_scales)}: {target_scales}), got {len(ratios)}."
            )
        invalid_ratios = [ratio for ratio in ratios if not 0 <= ratio < 1]
        if invalid_ratios:
            raise ValueError(f"sparsevar_pruning_ratios must be in [0, 1), got {invalid_ratios}.")
        return {scale: 1.0 - ratio for scale, ratio in zip(target_scales, ratios)}

    def sparsevar_should_prune(
            self,
            scale_index: int,
            repeat_index: int,
            infer_repeat_times: int,
            target_scales: Iterable[int],
    ) -> bool:
        if scale_index not in target_scales:
            return False
        if (
            bool(int(self.sparsevar_final_iteration_full))
            and infer_repeat_times > 1
            and repeat_index == infer_repeat_times - 1
        ):
            return False
        return True
