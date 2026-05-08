"""
==============================================================
Logger Module - Mobile Phone Detection System
==============================================================
Configurable logging with file and console output.
Provides structured logging for detection events,
API requests, and system diagnostics.
==============================================================
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "PhoneDetection",
    log_file: str = None,
    level: str = "INFO",
    log_format: str = None,
    date_format: str = None,
) -> logging.Logger:
    """
    Set up and configure a logger instance.

    Args:
        name: Logger name identifier
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        date_format: Custom date format string

    Returns:
        Configured logging.Logger instance
    """
    # Import config defaults
    from config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_FILE

    if log_format is None:
        log_format = LOG_FORMAT
    if date_format is None:
        date_format = LOG_DATE_FORMAT
    if log_file is None:
        log_file = str(LOG_FILE)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
