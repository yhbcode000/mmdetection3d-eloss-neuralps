# tools/logging_setup.py
import sys
import logging

def setup_logger():
    """Configures and returns a logger instance."""
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Use sys.stdout.flush to ensure logs are displayed immediately
    handler.flush = sys.stdout.flush
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger