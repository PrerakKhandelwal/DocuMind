"""
logger.py  — Centralised coloured logger.
"""
import logging
import sys
from pathlib import Path

import colorlog

from src.config_loader import cfg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = "%(log_color)s%(asctime)s [%(levelname)-8s] %(name)s%(reset)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    ch = colorlog.StreamHandler(sys.stdout)
    ch.setFormatter(colorlog.ColoredFormatter(fmt, datefmt=datefmt))
    ch.setLevel(level)
    logger.addHandler(ch)

    # File handler
    if cfg.logging.log_to_file:
        log_dir = Path(cfg.paths.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "agent.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s", datefmt=datefmt
        ))
        fh.setLevel(level)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
