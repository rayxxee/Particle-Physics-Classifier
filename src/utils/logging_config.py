"""
Structured logging configuration using structlog.

Usage:
    from src.utils.logging_config import get_logger
    log = get_logger(__name__)
    log.info("ETL started", dataset="higgs", n_samples=500_000)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from omegaconf import OmegaConf


def configure_logging(level: str = "INFO", format: str = "pretty") -> None:
    """
    Configure structlog for the entire application.

    Args:
        level:  Log level string — "DEBUG", "INFO", "WARNING", "ERROR".
        format: "pretty" for human-readable console output (dev),
                "json"   for machine-readable JSON output (prod).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Standard library logging setup (captures third-party logs)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors applied to every log record
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if format == "json":
        # JSON output — used in Docker / prod environments
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Pretty console output — used in dev
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Example:
        log = get_logger(__name__)
        log.info("Processing", n_events=42)
    """
    return structlog.get_logger(name)


def configure_from_config(cfg: Any) -> None:
    """
    Configure logging from an OmegaConf config object.

    Expects:
        cfg.logging.level  (str)
        cfg.logging.format (str)
    """
    level = OmegaConf.select(cfg, "logging.level", default="INFO")
    fmt = OmegaConf.select(cfg, "logging.format", default="pretty")
    configure_logging(level=level, format=fmt)
