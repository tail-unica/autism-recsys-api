import os

import numpy as np
import polars as pl
import torch
from hopwise.data.utils import PathLanguageModelingTokenType, create_dataset, data_preparation
from hopwise.model.logits_processor import LogitsProcessorList
from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from hopwise.utils import get_model, init_seed
from safetensors.torch import load_file
from transformers import AutoTokenizer  # , StoppingCriteriaList

from src.core.recommendation_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
    # ZeroShotCriteria,
    ZeroShotCumulativeSequenceScorePostProcessor,
)
from src.core.semantic_matcher import HierarchicalSemanticMatcher
from src.core.utils import cfg, logger

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load Data
data_dir = cfg.data.data_dir
food_data = pl.scan_parquet(os.path.join(data_dir, cfg.data.parquet_file))

# Initialize Semantic Matcher
all_food_semantic_matcher = HierarchicalSemanticMatcher(
    model_name=cfg.semantic_search.model_name,
    embedding_dim=cfg.semantic_search.embedding_dim,
    use_hierarchical=cfg.semantic_search.use_hierarchical,
    n_clusters=cfg.semantic_search.n_clusters,
    cache_dir=cfg.semantic_search.cache_dir,
    batch_size=cfg.semantic_search.batch_size,
)
all_food_semantic_matcher.encode_items(
    food_data.select(pl.col("name")).collect().to_numpy().flatten(),
    data_identifier="all_food_data",
    force_recompute=cfg.semantic_search.force_recompute,
)

ingredients_only_semantic_matcher = HierarchicalSemanticMatcher(
    model_name=cfg.semantic_search.model_name,
    embedding_dim=cfg.semantic_search.embedding_dim,
    use_hierarchical=cfg.semantic_search.use_hierarchical,
    n_clusters=cfg.semantic_search.n_clusters,
    cache_dir=cfg.semantic_search.cache_dir,
    batch_size=cfg.semantic_search.batch_size,
)
ingredients = np.unique(
    food_data.filter(pl.col("food_item_type") == "ingredient").select(pl.col("name")).collect().to_numpy().flatten()
)
ingredients_only_semantic_matcher.encode_items(
    ingredients,
    data_identifier="only_ingredients_data",
    force_recompute=cfg.semantic_search.force_recompute,
)

recipes_only_semantic_matcher = HierarchicalSemanticMatcher(
    model_name=cfg.semantic_search.model_name,
    embedding_dim=cfg.semantic_search.embedding_dim,
    use_hierarchical=cfg.semantic_search.use_hierarchical,
    n_clusters=cfg.semantic_search.n_clusters,
    cache_dir=cfg.semantic_search.cache_dir,
    batch_size=cfg.semantic_search.batch_size,
)
recipes = np.unique(
    food_data.filter(pl.col("food_item_type") == "recipe").select(pl.col("name")).collect().to_numpy().flatten()
)
recipes_only_semantic_matcher.encode_items(
    recipes,
    data_identifier="only_recipes_data",
    force_recompute=cfg.semantic_search.force_recompute,
)

logger.info("Loading checkpoint...")
checkpoint = torch.load(cfg.recommender.hopwise_checkpoint_file, weights_only=False)
config = checkpoint["config"]
config["checkpoint_dir"] = os.path.dirname(cfg.recommender.hopwise_checkpoint_file)
config["data_path"] = os.path.join(data_dir, cfg.data.recommender_dataset)
config["load_col"]["item"] = ["recipe_id", "name", "description"]
config._set_env_behavior()

logger.info("Initializing dataset...")
init_seed(config["seed"], config["reproducibility"])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

logger.info("Initializing recommender and applying checkpoint weights...")
recommender = get_model(config["model"])(config, train_data.dataset)
recommender = recommender.to(device=cfg.recommender.device, dtype=config["weight_precision"])

hf_checkpoint_file = cfg.recommender.hopwise_checkpoint_file.replace("hopwise", "huggingface")
weights = load_file(os.path.join(hf_checkpoint_file, "model.safetensors"))
recommender.load_state_dict(weights, strict=False)
dataset._tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_file)
logger.info("Compiling recommender for performance...")
recommender = torch.compile(recommender, mode=cfg.recommender.compile_mode)

logger.info("Creating post-processors and logits processors for recommendation generation...")
existing_user_cumulative_sequence_postprocessor = CumulativeSequenceScorePostProcessor(
    dataset.tokenizer, dataset.get_user_used_ids(), dataset.item_num
)
zero_shot_sequence_postprocessor = ZeroShotCumulativeSequenceScorePostProcessor(dataset.tokenizer, dataset.item_num)

constrained_logits_processors_list = recommender.logits_processor_list
zero_shot_restriction_logits_processor = RestrictionLogitsProcessorWordLevel(
    tokenized_ckg=dataset.get_tokenized_ckg(),
    tokenizer=dataset.tokenizer,
    propagate_connected_entities=cfg.recommender.propagate_connected_entities,
)

ui_relation = dataset.field2token_id[dataset.relation_field][dataset.ui_relation]
zero_shot_constrained_logits_processor = ZeroShotConstrainedLogitsProcessor(
    tokenized_ckg=dataset.get_tokenized_ckg(),
    tokenized_used_ids=dataset.get_tokenized_used_ids(),
    max_sequence_length=10,  # High as sequences for zero-shot should be shorter due to StoppingCriteria trigger
    tokenizer=dataset.tokenizer,
    remove_user_tokens_from_sequences=cfg.recommender.remove_user_tokens_from_sequences,
    tokenized_ui_relation=(
        dataset.tokenizer.convert_tokens_to_ids(PathLanguageModelingTokenType.RELATION.token + str(ui_relation))
    ),
)
zero_shot_constrained_logits_processors_list = LogitsProcessorList(
    [
        zero_shot_constrained_logits_processor,
        zero_shot_restriction_logits_processor,
    ]
)
# zero_shot_stop_criteria_list = StoppingCriteriaList([
#     ZeroShotCriteria(tokenizer=dataset.tokenizer)
# ])

kg_elements_semantic_matcher = HierarchicalSemanticMatcher(
    model_name=cfg.semantic_search.model_name,
    embedding_dim=cfg.semantic_search.embedding_dim,
    use_hierarchical=cfg.semantic_search.use_hierarchical,
    n_clusters=cfg.semantic_search.n_clusters,
    cache_dir=cfg.semantic_search.cache_dir,
    batch_size=cfg.semantic_search.batch_size,
)

non_items_kg_elements = dataset.field2id_token[dataset.entity_field][dataset.item_num :]  # skip items ids
items_kg_elements = list(dataset.entity2item.keys())
kg_elements = np.concatenate([items_kg_elements, non_items_kg_elements])
no_id_kg_elements_map = {}
for el in kg_elements:
    name = ".".join(el.split(".", maxsplit=2)[::2])  # skips numeric ID
    no_id_kg_elements_map.setdefault(name, []).append(el)

kg_elements_semantic_matcher.encode_items(
    np.array(list(no_id_kg_elements_map.keys())),
    data_identifier="kg_elements_data",
    force_recompute=cfg.semantic_search.force_recompute,
)
