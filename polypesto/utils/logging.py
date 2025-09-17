"""Logging utilities for capturing and redirecting output."""

import sys
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TextIO


@contextmanager
def redirect_output_to_file(
    log_file: str | Path,
    stdout: bool = True,
    stderr: bool = True,
    capture_logging: bool = True,
    mode: str = "w",
):
    """
    Context manager to redirect stdout/stderr and logging to a log file.

    Parameters
    ----------
    log_file : Path
        Path to the log file
    stdout : bool
        Whether to redirect stdout (default: True)
    stderr : bool
        Whether to redirect stderr (default: True)
    capture_logging : bool
        Whether to capture Python logging output (default: True)
    mode : str
        File open mode ('w' for overwrite, 'a' for append)

    Example
    -------
    >>> with redirect_output_to_file(Path("compilation.log")):
    ...     print("This goes to the log file")
    ...     logging.info("This also goes to the log file")
    ...     some_verbose_function()
    """

    log_file = Path(log_file)

    original_stdout = sys.stdout if stdout else None
    original_stderr = sys.stderr if stderr else None

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Setup logging capture
    file_handler = None
    all_loggers = []
    original_handlers = {}
    original_levels = {}

    if capture_logging:
        # Store original state of ALL loggers
        all_loggers = [
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ]
        all_loggers.append(logging.getLogger())  # Add root logger

        original_handlers = {}
        original_levels = {}
        original_propagate = {}

        for logger in all_loggers:
            original_handlers[logger.name] = logger.handlers[:]
            original_levels[logger.name] = logger.level
            original_propagate[logger.name] = logger.propagate

        # Create file handler for logging
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Replace ALL logger handlers with our file handler
        for logger in all_loggers:
            logger.handlers.clear()
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
            # Disable propagation to prevent duplicate messages
            logger.propagate = False

    with open(log_file, mode) as f:
        try:
            if stdout:
                sys.stdout = f
            if stderr:
                sys.stderr = f
            yield f
        finally:
            # Restore stdout/stderr
            if original_stdout:
                sys.stdout = original_stdout
            if original_stderr:
                sys.stderr = original_stderr

            # Restore logging for ALL loggers
            if capture_logging and file_handler:
                for logger in all_loggers:
                    logger.handlers.clear()
                    logger.handlers.extend(original_handlers.get(logger.name, []))
                    logger.setLevel(original_levels.get(logger.name, logging.WARNING))
                    logger.propagate = original_propagate.get(logger.name, True)
                file_handler.close()


def get_log_file_path(
    base_name: str, log_dir: Optional[Path] = None, suffix: str = ".log"
) -> Path:
    """
    Generate a standardized log file path.

    Parameters
    ----------
    base_name : str
        Base name for the log file (e.g., model name)
    log_dir : Optional[Path]
        Directory for log files (defaults to ./logs)
    suffix : str
        File suffix (default: ".log")

    Returns
    -------
    Path
        Full path to the log file
    """
    if log_dir is None:
        log_dir = Path("logs")

    return log_dir / f"{base_name}{suffix}"
