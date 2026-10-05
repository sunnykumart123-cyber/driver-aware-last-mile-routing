from __future__ import annotations

import logging
import os

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("MTP_LOGLEVEL", "INFO").upper())
        logger.propagate = False
    return logger
