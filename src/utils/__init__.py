"""Utilities package for the Particle Physics Classifier."""

from src.utils.logging_config import get_logger, configure_logging, configure_from_config

# Device utils use torch — import lazily to avoid blocking tests in environments
# where torch is not yet installed.
def get_device(*args, **kwargs):  # noqa: E302
    from src.utils.device_utils import get_device as _get_device
    return _get_device(*args, **kwargs)


def get_amp_context(*args, **kwargs):
    from src.utils.device_utils import get_amp_context as _ctx
    return _ctx(*args, **kwargs)


def get_amp_scaler(*args, **kwargs):
    from src.utils.device_utils import get_amp_scaler as _scaler
    return _scaler(*args, **kwargs)


def device_info(*args, **kwargs):
    from src.utils.device_utils import device_info as _info
    return _info(*args, **kwargs)


__all__ = [
    "get_device",
    "get_amp_context",
    "get_amp_scaler",
    "device_info",
    "get_logger",
    "configure_logging",
    "configure_from_config",
]
