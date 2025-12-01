import logging

def get_logger(name: str = "magi") -> logging.Logger:
    """Get a logger for MAGI model operations."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Add console handler if no handlers exist
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

# Create the default magi_logger instance
magi_logger = get_logger()