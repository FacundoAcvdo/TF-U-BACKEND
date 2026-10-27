import logging
import os
import sys

_INITIALIZED: set[str] = set()


def setup_service_logger(service_name: str) -> logging.Logger:
    logger = logging.getLogger(service_name)
    if service_name in _INITIALIZED:
        return logger
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [" + service_name + "] %(message)s"
        )
    )
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    _INITIALIZED.add(service_name)
    return logger
