"""Helpers to parse inference args for the T2I evaluation scripts.

Model-level args live in Tap classes (InfinityInferArgs / HARTInferArgs in
utils/arg_util.py) and are dispatched on --model_type; benchmark-level args
stay in each eval script's own argparse parser and are parsed from the argv
tokens left over by Tap (known_only=True).

Typical usage in an eval script::

    args = parse_infer_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=4)
    ...
    args = merge_script_args(args, parser)

For video (InfinityStar-family) entry scripts, `parse_video_infer_args` plays
the same dispatch role over the utils/arg_util_video.InferArgs subclasses.
"""

import sys
from importlib import import_module

from utils.arg_util import HARTInferArgs, InfinityInferArgs


def _peek_model_type(argv, default='infinity_2b'):
    """Read --model_type from raw argv without consuming it."""
    for i, tok in enumerate(argv):
        if tok == '--model_type' and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith('--model_type='):
            return tok.split('=', 1)[1]
    return default


def parse_infer_args(argv=None):
    """Parse model-level inference args, dispatching the Tap class on --model_type.

    Unrecognized (benchmark-level) tokens are kept in ``args.extra_args`` for
    ``merge_script_args`` to consume.
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # drop the legacy launcher-injected rank flag (torchrun uses the LOCAL_RANK env var)
    argv = [tok for tok in argv if not tok.startswith(('--local-rank=', '--local_rank='))]

    model_type = _peek_model_type(argv)
    args_cls = HARTInferArgs if 'hart' in model_type else InfinityInferArgs
    return args_cls(explicit_bool=True).parse_args(args=argv, known_only=True)


# model_type -> (args module, Tap class) for the video (InfinityStar-family) models;
# mirrors EXACT_MODEL_MODULES in models/registry.py. Kept as import strings so arg
# parsing stays lightweight: the args modules only import utils.arg_util_video and
# the heavy model modules are loaded later by the registry.
VIDEO_INFER_ARGS_REGISTRY = {
    'faststar_infinitystar': ('models.faststar.args_faststar', 'FastStarArgs'),
    'fastvar_infinitystar': ('models.fastvar.args_fastvar', 'FastvarArgs'),
    'sparsevar_infinitystar': ('models.sparsevar.args_sparsevar', 'SparsevarArgs'),
    # infinitystar_qwen8b / sparvar_infinitystar use the plain InferArgs (fallback)
}


def parse_video_infer_args(argv=None):
    """Parse video inference args, dispatching the InferArgs subclass on --model_type.

    The accelerated model classes (FastVAR / FastSTAR / SparseVAR) read their
    method-specific fields and helper methods from ``self.other_args`` during
    ``__init__`` and inference, so entry scripts (VBench, latency profiling, ...)
    must construct the matching XxxArgs subclass rather than the plain InferArgs.

    Unlike the T2I ``parse_infer_args``, parsing is strict (no ``known_only``):
    the video entry scripts keep all benchmark-level fields on the InferArgs
    classes themselves, so unknown flags should fail loudly.
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # drop the legacy launcher-injected rank flag (torchrun uses the LOCAL_RANK env var)
    argv = [tok for tok in argv if not tok.startswith(('--local-rank=', '--local_rank='))]

    model_type = _peek_model_type(argv, default='')
    entry = VIDEO_INFER_ARGS_REGISTRY.get(model_type)
    if entry is None:
        from utils.arg_util_video import InferArgs as args_cls
    else:
        module_name, cls_name = entry
        args_cls = getattr(import_module(module_name), cls_name)
    return args_cls().parse_args(args=argv)


def merge_script_args(model_args, parser):
    """Parse benchmark-level args from ``model_args.extra_args`` and merge them in.

    Raises if the script parser re-defines a model-level field, which would
    otherwise silently clobber the value Tap already parsed.
    """
    script_args = parser.parse_args(model_args.extra_args)
    for k, v in vars(script_args).items():
        if k in model_args.class_variables:
            raise ValueError(
                f"benchmark arg '--{k}' duplicates a model-level arg of "
                f"{type(model_args).__name__}; define it only in the Tap infer args class"
            )
        setattr(model_args, k, v)
    return model_args
