"""Internal logging abstraction wrapping the Loguru backend."""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(log_file: Path | None = None, debug_mode: bool = False) -> None:
    """Configures the unified logging backend outputs.

    Args:
        log_file: Optional filepath to write rolling debug logs to.
        debug_mode: If True, prints verbose debug logs to stdout.
    """
    # Remove default handler
    logger.remove()

    # Add styled stdout console handler
    console_level = "DEBUG" if debug_mode else "INFO"
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=console_level,
        colorize=True,
    )

    # Add rotating file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            rotation="10 MB",
            retention="10 days",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} - {message}"
            ),
            level="DEBUG",
            encoding="utf-8",
        )


def info(msg: str, *args: object, **kwargs: object) -> None:
    """Logs an info message."""
    logger.info(msg, *args, **kwargs)


def debug(msg: str, *args: object, **kwargs: object) -> None:
    """Logs a debug message."""
    logger.debug(msg, *args, **kwargs)


def warning(msg: str, *args: object, **kwargs: object) -> None:
    """Logs a warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args: object, **kwargs: object) -> None:
    """Logs an error message."""
    logger.error(msg, *args, **kwargs)


def exception(msg: str, *args: object, **kwargs: object) -> None:
    """Logs an exception traceback with a message."""
    logger.exception(msg, *args, **kwargs)
