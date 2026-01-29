from __future__ import annotations

import datetime
import logging
import os
from functools import lru_cache

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

CONFIG_PATH = os.path.join(os.pardir, os.pardir, "config")  # src/ -> config/

@lru_cache(maxsize=1)
def get_cfg():
    # Initialize Hydra only once
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path=CONFIG_PATH, version_base=None):
        return compose(config_name="default")

class CustomFormatter(logging.Formatter):
    # Use standard ANSI color codes to ensure terminal support
    grey = "\x1b[90m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str, datefmt: str | None = None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._level_formats = {
            logging.DEBUG: f"{self.grey}{fmt}{self.reset}",
            logging.INFO: fmt,
            logging.WARNING: f"{self.yellow}{fmt}{self.reset}",
            logging.ERROR: f"{self.red}{fmt}{self.reset}",
            logging.CRITICAL: f"{self.bold_red}{fmt}{self.reset}",
        }

    def format(self, record):
        record.levelname_c = f"{record.levelname}:"
        log_fmt = self._level_formats.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt, self.datefmt)
        return formatter.format(record)

@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    cfg = get_cfg()
    logfile = os.path.join("logs", f"core-{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    logger = logging.getLogger("AutismRecsysAPI")
    logger.setLevel(cfg.logging.level)

    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(CustomFormatter(cfg.logging.format))

        sh = logging.StreamHandler()
        sh.setLevel(cfg.logging.level)
        sh.setFormatter(CustomFormatter(cfg.logging.format))

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger