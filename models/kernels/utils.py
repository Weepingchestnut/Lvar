# -*- coding: utf-8 -*-

import contextlib
import functools
import logging
import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
import triton
import transformers
from packaging import version

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System
    if sys.platform == 'win32':
        logger.warning(
            "Detected Windows operating system. Triton does not have an official Windows release, "
            "thus FLA will not be adapted for Windows, and any potential errors will not be fixed. "
            "Please consider using a Linux environment for compatibility."
        )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.2.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.2.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton."
        )

    # Check Python version
    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience."
        )

    return None


check_environments()


def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            import warnings
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def input_guard(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(ctx,
                      *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                      **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
        return wrapper
    return decorator


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


def _cpu_device_warning():
    import warnings
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
    except BaseException:
        _cpu_device_warning()
        return -1


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'


@lru_cache(maxsize=None)
def _check_platform() -> Literal['nvidia', 'amd', 'intel', 'musa']:
    device = get_available_device()
    if device == 'cuda':
        return 'nvidia'
    elif device == 'hip':
        return 'amd'
    elif device == 'xpu':
        return 'intel'
    else:
        return device


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = _check_platform()

is_amd = (device_platform == 'amd')
is_intel = (device_platform == 'intel')
is_nvidia = (device_platform == 'nvidia')
is_intel_alchemist = (is_intel and 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0))
is_nvidia_hopper = (is_nvidia and ('NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9))
use_cuda_graph = (is_nvidia and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8)
is_gather_supported = hasattr(triton.language, 'gather')


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)


class Cache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    is_compileable = True

    def __init__(
        self,
        seen_tokens: int = 0
    ):
        super().__init__()

        self.states: List[Dict[str, Any]] = []

        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += offset

        if cache_kwargs is None:
            cache_kwargs = {}
        if attn_state is not None:
            input_size = attn_state[0].shape[1]
            window_size = cache_kwargs.get('window_size', None)
            if not isinstance(attn_state, Tuple):
                raise ValueError("`attn_state` must be a tuple of tensors for key/value states")
        if len(self.states) <= layer_idx:
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = [state[:, -window_size:].contiguous() for state in attn_state]
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state
            )
            self.states.append(state)
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                if window_size is not None and state['attn_state'][0].shape[1] == window_size:
                    for i, (old_state, new_state) in enumerate(zip(state['attn_state'], attn_state)):
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        old_state = old_state.roll(-input_size, 1)
                        # replace the last `input_size` tokens with the new key/value states
                        old_state[:, -input_size:] = new_state
                        state['attn_state'][i] = old_state
                else:
                    attn_state = [
                        torch.cat([old_state, new_state], 1)
                        for old_state, new_state in zip(state['attn_state'], attn_state)
                    ]
                state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        seen_tokens: int = 0
    ):
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(seen_tokens)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache

