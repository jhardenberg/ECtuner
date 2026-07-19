"""
Logging Configuration Module.

Provides a centralized and flexible logging setup for ECtuner, supporting 
both console and file-based logging with automatic level resolution.
"""
import logging
from typing import Optional, Union


def setup_logger(
    level: Optional[Union[str, int]] = None, 
    name: str = 'ectuner', 
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Defines and configures a logger for ECtuner.

    Prevents handler duplication if called multiple times and sets up 
    both terminal stream and file handlers if requested.

    Args:
        level (str | int, optional): The desired logging level 
            (e.g., 'INFO', 'DEBUG', logging.INFO). Defaults to None (WARNING).
        name (str, optional): The name of the logger instance. Defaults to 'ectuner'.
        log_file (str, optional): Path to the output log file. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.
    """
    loglev = convert_logger(level)
    logger = logging.getLogger(name)

    # Update level if logger already exists with handlers
    if logger.handlers:
        if loglev != logging.getLevelName(logger.getEffectiveLevel()):
            logger.setLevel(loglev)
            logger.info('Updating the log_level to %s', loglev)
        return logger

    # Avoid duplication/propagation of loggers to the root logger
    logger.propagate = False
    logger.setLevel(loglev)

    # Create a formatter for the console output
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)8s -> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Stream Handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Optional File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger


def convert_logger(loglev: Optional[Union[str, int]] = None) -> str:
    """
    Converts a string or integer to a valid logging level string.

    Args:
        loglev (str | int | None): The input log level.

    Returns:
        str: The sanitized standard logging level name (e.g., 'INFO').
        
    Raises:
        ValueError: If the input type is neither string, integer, nor None.
    """
    loglev_default = "WARNING"

    if isinstance(loglev, str):
        loglev = loglev.upper()
    elif isinstance(loglev, int):
        loglev = logging.getLevelName(loglev)
    elif loglev is None:
        loglev = loglev_default
    else:
        raise ValueError('Invalid log level type. Must be a string or an integer.')

    # Check if the log level exists in the logging module
    loglev_int = getattr(logging, loglev, None)

    if loglev_int is None:
        logging.warning("Invalid logging level '%s' specified. Setting it back to default '%s'.",
                        loglev, loglev_default)
        loglev = loglev_default

    return loglev