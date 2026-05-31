import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_MODULE_LOADING",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "OMP_NUM_THREADS",
)

DEFAULT_PIP_PATTERNS = (
    "accelerate",
    "decord",
    "diffusers",
    "einops",
    "flash",
    "huggingface",
    "imageio",
    "lightning",
    "numpy",
    "opencv",
    "pillow",
    "torch",
    "transformers",
    "triton",
    "xformers",
)


def _run_command(
    command: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: int = 15,
) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _format_section(title: str, body: Optional[str]) -> str:
    if body is None or len(body.strip()) == 0:
        body = "Could not collect"
    return f"=== {title} ===\n{body.strip()}"


def get_git_report(repo_root: Optional[Path] = None) -> str:
    repo_root = repo_root or _repo_root()
    branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run_command(["git", "status", "--short"], cwd=repo_root)
    if status is None:
        dirty = "Could not collect"
    else:
        dirty = "Yes" if status else "No"

    lines = [
        f"repo_root: {repo_root}",
        f"branch: {branch or 'Could not collect'}",
        f"commit: {commit or 'Could not collect'}",
        f"dirty: {dirty}",
    ]
    return "\n".join(lines)


def get_environment_variables_report(keys: Iterable[str] = DEFAULT_ENV_KEYS) -> str:
    lines = []
    for key in keys:
        value = os.environ.get(key)
        if value:
            lines.append(f"{key}={value}")
    return "\n".join(lines) if lines else "No selected environment variables are set"


def get_nvidia_smi_report() -> str:
    query_output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        timeout=10,
    )
    if query_output is not None:
        return "\n".join(
            f"gpu[{line.strip()}]" for line in query_output.splitlines() if line.strip()
        )

    full_output = _run_command(["nvidia-smi"], timeout=10)
    return full_output or "Could not collect"


def get_pip_packages_report(patterns: Iterable[str] = DEFAULT_PIP_PATTERNS) -> str:
    output = _run_command(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        timeout=20,
    )
    if output is None:
        return "Could not collect"

    lowered_patterns = tuple(pattern.lower() for pattern in patterns)
    lines = [
        line
        for line in output.splitlines()
        if any(pattern in line.lower() for pattern in lowered_patterns)
    ]
    return "\n".join(lines) if lines else "No matching packages"


def get_torch_collect_env_report() -> str:
    try:
        from torch.utils.collect_env import get_pretty_env_info
    except Exception as exc:
        return f"Could not import torch.utils.collect_env: {exc}"

    try:
        return get_pretty_env_info()
    except Exception as exc:
        return f"Could not collect torch environment: {exc}"


def get_lvar_env_report(
    include_torch_collect_env: bool = True,
    include_pip_packages: bool = True,
    include_nvidia_smi: bool = True,
) -> str:
    sections = [
        _format_section("Lvar Git", get_git_report()),
        _format_section("Selected Environment Variables", get_environment_variables_report()),
    ]
    if include_nvidia_smi:
        sections.append(_format_section("NVIDIA SMI", get_nvidia_smi_report()))
    if include_pip_packages:
        sections.append(_format_section("Selected Python Packages", get_pip_packages_report()))
    if include_torch_collect_env:
        sections.append(_format_section("PyTorch Collect Env", get_torch_collect_env_report()))
    return "\n\n".join(sections)


def print_lvar_env_report(
    include_torch_collect_env: bool = True,
    include_pip_packages: bool = True,
    include_nvidia_smi: bool = True,
) -> None:
    print(get_lvar_env_report(
        include_torch_collect_env=include_torch_collect_env,
        include_pip_packages=include_pip_packages,
        include_nvidia_smi=include_nvidia_smi,
    ))
