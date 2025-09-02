import datetime
import logging
import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
with initialize(config_path=os.path.join(os.pardir, os.pardir, "config"), version_base=None):
    cfg = compose(config_name="default")


def config_logger():
    # Configure logging
    logfile = os.path.join("logs", f"core-{datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')}.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(cfg.logging.format))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(cfg.logging.level)
    stream_handler.setFormatter(logging.Formatter(cfg.logging.format))

    logging.basicConfig(level=cfg.logging.level, handlers=[file_handler, stream_handler])
    return logging.getLogger("PHaSE API")


logger = config_logger()
