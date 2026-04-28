"""Unified logging. Rich color to terminal, plain text to file."""
import logging

from rich.console import Console
from rich.logging import RichHandler

from src.config import LOG_DIR, LOG_LEVEL

console = Console()


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if logger.handlers:
        return logger

    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.addHandler(rich_handler)

    if log_file:
        file_handler = logging.FileHandler(LOG_DIR / log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
