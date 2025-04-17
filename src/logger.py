# src/logger.py
# src/logger.py
import logging
import os
import datetime
from logging.handlers import RotatingFileHandler

class Logger:
    """
    A class to set up and manage logging for the Iris Classification project.
    """
    def __init__(self, log_level=logging.INFO, log_dir='logs'):
        """
        Initialize the logger with specified log level and directory.
        
        Args:
            log_level: The logging level (default: logging.INFO)
            log_dir: Directory to store log files (default: 'logs')
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamp for the log file name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'iris_classification_{timestamp}.log')
        
        # Configure the root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers to avoid duplicate logs
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Set up file handler with rotation (max 10MB, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def get_logger(self):
        """Return the configured logger."""
        return self.logger

# Convenience function to get a configured logger
def setup_logger(name=None, log_level=logging.INFO):
    """
    Set up and return a logger with the specified name and level.
    
    Args:
        name: Logger name (optional)
        log_level: The logging level (default: logging.INFO)
    
    Returns:
        A configured logger instance
    """
    # Initialize the main logger if not already done
    if not logging.getLogger().handlers:
        Logger(log_level=log_level)
    
    # Return either the root logger or a named logger
    if name:
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        return logger
    else:
        return logging.getLogger()