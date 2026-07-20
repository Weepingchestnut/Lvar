from importlib import import_module
from timm.models import is_model


EXACT_MODEL_MODULES = {
    "sparvar_infinitystar": "models.sparvar.sparvar_infinitystar_model",
    "faststar_infinitystar": "models.faststar.faststar_model",
    "fastvar_infinitystar": "models.fastvar.fastvar_model_infinitystar",
    "sparsevar_infinitystar": "models.sparsevar.sparsevar_model_infinitystar",
}

PREFIX_MODEL_MODULES = {
    # --- FastVAR (ICCV'2025) ---
    "fastvar_infinity_": "models.fastvar.fastvar_model",
    # --- SparseVAR (ICCV'2025) ---
    "sparsevar_infinity_": "models.sparsevar.sparsevar_model",
    # --- SkipVAR (arXiv:2506)---
    "skipvar_infinity_": "models.skipvar.skipvar_model",
    # "infinitystar_": "models.infinitystar.infinitystar_model",
    # "infinity_": "models.infinity.infinity_model",
    "sparvar_infinity_": "models.sparvar.sparvar_model",
    # "scalekv_infinity_": "models.infinity.infinity_model",
}


def ensure_model_registered(model_type: str) -> None:
    if is_model(model_type):
        return

    module_name = EXACT_MODEL_MODULES.get(model_type)
    if module_name is None:
        module_name = next(
            (
                module
                for prefix, module in PREFIX_MODEL_MODULES.items()
                if model_type.startswith(prefix)
            ),
            None,
        )

    if module_name is not None:
        import_module(module_name)

    if not is_model(model_type):
        raise RuntimeError(f"Unknown model after registry import: {model_type}")
