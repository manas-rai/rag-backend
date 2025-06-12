"""
This module provides a logger setup for the application.
It allows for logging to a file and to the console.
It also allows for logging to a specific logger name.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO
        ) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.
    
    Args:
        name: The name of the logger
        log_file: Optional path to log file. If None, logs to stdout
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create handlers
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create default logger for the application
app_logger = setup_logger('rag-backend')
