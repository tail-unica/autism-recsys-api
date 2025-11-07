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
        fh.setFormatter(logging.Formatter(cfg.logging.format))

        sh = logging.StreamHandler()
        sh.setLevel(cfg.logging.level)
        sh.setFormatter(logging.Formatter(cfg.logging.format))

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger