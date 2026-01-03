"""
Logging utilities for Ghost-1
Provides structured logging with session tracking and rotation
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

class JsonFormatter(logging.Formatter):
    """Formatter to output JSON structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(
    session_id: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_json: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging with both console and file handlers.
    
    Args:
        session_id: Unique session identifier (auto-generated if None)
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_json: Use JSON formatting for file logs
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Root logger configured with handlers
    """
    # Create logs directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Generate session ID if not provided
    if session_id is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (with rotation)
    log_file = Path(log_dir) / f"ghost_{session_id}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Capture all levels to file
    
    if enable_json:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        ))
    
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Session: {session_id}, Level: {log_level}, File: {log_file}")
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
