"""Structured logging setup for the MedVQA project.

Provides consistent logging across training, evaluation, and inference.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "medvqa",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    verbose: bool = False,
) -> logging.Logger:
    """Set up a structured logger with consistent formatting.

    Args:
        name: Logger name (usually __name__).
        log_file: Optional path to a log file.
        level: Logging level (default: INFO).
        verbose: If True, set level to DEBUG.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else level)

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "medvqa") -> logging.Logger:
    """Get an existing logger or create a default one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
