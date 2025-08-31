"""
Logging utilities for the sensitive word filtering system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from config.settings import config

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Log file path (optional)
        level: Logging level
        format_string: Custom format string (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Format
    if not format_string:
        format_string = config.logging_config.get(
            "format", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if not log_file:
        log_file = config.logging_config.get("file")
    
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        max_bytes = _parse_size(config.logging_config.get("max_size", "10MB"))
        backup_count = config.logging_config.get("backup_count", 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def _parse_size(size_str: str) -> int:
    """Parse size string like '10MB' to bytes.
    
    Args:
        size_str: Size string (e.g., '10MB', '5GB')
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3
    }
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            number = size_str[:-len(suffix)]
            try:
                return int(float(number) * multiplier)
            except ValueError:
                pass
    
    # Default to 10MB if parsing fails
    return 10 * 1024 ** 2

# Global logger instances
main_logger = setup_logger("sensitive_filter")
training_logger = setup_logger("training")
inference_logger = setup_logger("inference")
