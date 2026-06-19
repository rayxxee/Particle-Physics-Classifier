"""
Device utilities: auto-detect best available compute device and configure AMP.

Priority order: CUDA > MPS (Apple Silicon) > CPU.

All torch imports are lazy (inside functions) so that this module can be
imported without torch installed — useful for pure-Python tests and CI
environments that install a lightweight subset of dependencies.

Usage:
    from src.utils.device_utils import get_device, get_amp_context

    device = get_device()
    with get_amp_context(device) as ctx:
        outputs = model(inputs.to(device))
"""

from __future__ import annotations

import contextlib
from typing import Any, Generator


def get_device(prefer: str | None = None) -> Any:
    """
    Return the best available torch device.

    Args:
        prefer: Force a specific device string ("cuda", "mps", "cpu").
                If None, auto-detects in priority order CUDA > MPS > CPU.

    Returns:
        torch.device instance.
    """
    import torch

    if prefer is not None:
        device = torch.device(prefer)
        _validate_device(device)
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon GPU via Metal Performance Shaders
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _validate_device(device: Any) -> None:
    """Raise ValueError if requested device is not available."""
    import torch
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is not available.")
    if device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS device requested but MPS is not available.")


def is_amp_supported(device: Any) -> bool:
    """
    Return True if Automatic Mixed Precision (AMP) is supported on the device.

    AMP is only supported on CUDA. MPS has partial support but is unstable.
    CPU AMP is supported in PyTorch but gives no speed benefit.
    """
    return device.type == "cuda"


def get_amp_scaler(device: Any) -> Any:
    """
    Return a GradScaler for AMP training, or None if AMP is not supported.

    The scaler must be used in the training loop:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    If None is returned, use normal loss.backward() + optimizer.step().
    """
    if is_amp_supported(device):
        from torch.cuda.amp import GradScaler
        return GradScaler()
    return None


@contextlib.contextmanager
def get_amp_context(device: Any) -> Generator:
    """
    Context manager that enables AMP autocasting on CUDA, no-op otherwise.

    Usage:
        with get_amp_context(device):
            outputs = model(inputs)
    """
    import torch
    if is_amp_supported(device):
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def device_info(device: Any) -> dict:
    """Return a dict of device metadata for logging."""
    import torch
    info: dict = {"device": str(device), "amp_supported": is_amp_supported(device)}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1e9, 2
        )
        info["cuda_version"] = torch.version.cuda
    return info
