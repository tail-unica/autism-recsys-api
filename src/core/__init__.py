import datetime
import logging
import os

import polars as pl
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.core.semantic_matcher import HierarchicalSemanticMatcher

script_dir = os.path.dirname(os.path.abspath(__file__))

GlobalHydra.instance().clear()
with initialize(config_path=os.path.join(os.pardir, os.pardir, "config"), version_base=None):
    cfg = compose(config_name="default")

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

# Load Data
data_dir = cfg.data.data_dir
food_data = pl.scan_parquet(os.path.join(data_dir, cfg.data.parquet_file))

# Initialize Semantic Matcher
semantic_matcher = HierarchicalSemanticMatcher(
    model_name=cfg.semantic_search.model_name,
    embedding_dim=cfg.semantic_search.embedding_dim,
    use_hierarchical=cfg.semantic_search.use_hierarchical,
    n_clusters=cfg.semantic_search.n_clusters,
    cache_dir=cfg.semantic_search.cache_dir,
    batch_size=cfg.semantic_search.batch_size,
)
semantic_matcher.encode_food_items(
    food_data.select(pl.col("name")).collect().to_numpy().flatten(),
    force_recompute=cfg.semantic_search.force_recompute,
)
