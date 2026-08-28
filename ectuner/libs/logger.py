"""
Logging Configuration Module.

Provides a centralized and flexible logging setup for ECtuner, supporting 
both console and file-based logging with automatic level resolution.
"""
import logging
import os
from typing import Optional, Union


def setup_logger(
    level: Optional[Union[str, int]] = None, 
    name: str = 'ectuner', 
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Defines and configures a logger for ECtuner.

    Prevents general handler duplication if called multiple times, but intelligently 
    swaps file handlers if the execution loop requests a different log file path.

    Args:
        level: The desired logging level (e.g., 'INFO', 'DEBUG', logging.INFO). 
            Defaults to WARNING if None.
        name: The name of the logger instance.
        log_file: Path to the output log file.

    Returns:
        The configured logger instance.
    """
    loglev = convert_logger(level)
    logger = logging.getLogger(name)
    logger.propagate = False

    # Update level if needed
    if logger.handlers and loglev != logging.getLevelName(logger.getEffectiveLevel()):
        logger.setLevel(loglev)
        logger.info('Updating the log_level to %s', loglev)
    else:
        logger.setLevel(loglev)

    # Check if a StreamHandler already exists
    has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
    
    if not has_stream_handler:
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)8s -> %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Intelligently swap FileHandler if requested
    if log_file:
        abs_log_path = os.path.abspath(log_file)
        
        existing_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        file_handler_exists = False
        
        for h in existing_file_handlers:
            if h.baseFilename == abs_log_path:
                file_handler_exists = True
            else:
                # Remove and safely close old handlers to prevent log leaking across loop iterations
                logger.removeHandler(h)
                h.close()
                
        if not file_handler_exists:
            file_handler = logging.FileHandler(abs_log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)

    return logger


def convert_logger(loglev: Optional[Union[str, int]] = None) -> str:
    """
    Converts a string or integer to a valid logging level string.

    Args:
        loglev: The input log level.

    Returns:
        The sanitized standard logging level name (e.g., 'INFO').
        
    Raises:
        ValueError: If the input type is neither string, integer, nor None.
    """
    loglev_default = "WARNING"

    if isinstance(loglev, str):
        loglev = loglev.upper()
    elif isinstance(loglev, int):
        loglev = logging.getLevelName(loglev)
    elif loglev is None:
        return loglev_default
    else:
        raise ValueError('Invalid log level type. Must be a string or an integer.')

    # Check if the log level exists in the logging module
    if getattr(logging, loglev, None) is None:
        return loglev_default

    return loglev