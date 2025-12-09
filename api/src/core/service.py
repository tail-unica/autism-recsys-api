from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from neo4j import GraphDatabase
import os
import numpy as np

from src.core.utils import get_cfg, get_logger
from src.core.dummy import dummy_place_info_fetcher, dummy_place_recommender
from src.core.info import fetch_place_info
from src.core.data import load_data

import torch
from hopwise.utils import get_model, init_seed
from hopwise.data.utils import PathLanguageModelingTokenType, create_dataset, data_preparation
from hopwise.model.logits_processor import LogitsProcessorList
from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.core.setup_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
    ZeroShotCumulativeSequenceScorePostProcessor,
)

class RecommenderService:
    def __init__(self) -> None:
        self._cfg = None
        self._logger = None
        self._ready = False
        self._warmup_task: Optional[asyncio.Task] = None

        self._model = None
        self._tokenizer = None
        self._indices = None
        self._matcher = None

        self._neo4j_driver = None

    async def initialize(self) -> None:
        self._cfg = get_cfg()
        self._logger = get_logger()

        # --- Neo4j Setup ---
        self._neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )
        self._logger.info("Connected to Neo4j database.")

        # --- Recommender Model Setup ---
        self._logger.info("RecommenderService: Starting initialization.")

        self._logger.info("Loading checkpoint...")
        checkpoint = torch.load(
            self._cfg.model.hopwise_checkpoint_file, 
            map_location=self._cfg.model.device, 
            weights_only=False,
        )
        config = checkpoint["config"]
        config["checkpoint_dir"] = os.path.dirname(self._cfg.model.hopwise_checkpoint_file)

        self._logger.info("Loading dataset...")
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        with self._neo4j_driver.session() as session:
            load_data(session, data_dir, dataset=self._cfg.data.dataset)
        config["data_path"] = os.path.join(data_dir, self._cfg.data.dataset)
        config["load_col"]["item"] = ["poi_id", "name"]
        config._set_env_behavior()

        self._logger.info("Initializing dataset...")
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        self._logger.info("Initializing recommender and applying checkpoint weights...")
        recommender = get_model(config["model"])(config, train_data.dataset)
        recommender = recommender.to(device=self._cfg.model.device, dtype=config["weight_precision"])

        # --- checkpoint weights ----
        hf_checkpoint_file = self._cfg.model.hopwise_checkpoint_file.replace("hopwise", "huggingface")
        weights = load_file(os.path.join(hf_checkpoint_file, "model.safetensors"))
        recommender.load_state_dict(weights, strict=False)
        dataset._tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_file)

        self._logger.info("Compiling recommender for performance...")
        recommender = torch.compile(recommender, mode=self._cfg.model.compile_mode)

        self._logger.info("Creating post-processors and logits processors for recommendation generation...")
        existing_user_cumulative_sequence_postprocessor = CumulativeSequenceScorePostProcessor(
            dataset.tokenizer, dataset.get_user_used_ids(), dataset.item_num
        )
        zero_shot_sequence_postprocessor = ZeroShotCumulativeSequenceScorePostProcessor(dataset.tokenizer, dataset.item_num)

        constrained_logits_processors_list = recommender.logits_processor_list
        zero_shot_restriction_logits_processor = RestrictionLogitsProcessorWordLevel(
            tokenized_ckg=dataset.get_tokenized_ckg(),
            tokenizer=dataset.tokenizer,
            propagate_connected_entities=self._cfg.model.propagate_connected_entities,
        )

        ui_relation = dataset.field2token_id[dataset.relation_field][dataset.ui_relation]
        zero_shot_constrained_logits_processor = ZeroShotConstrainedLogitsProcessor(
            tokenized_ckg=dataset.get_tokenized_ckg(),
            tokenized_used_ids=dataset.get_tokenized_used_ids(),
            max_sequence_length=10,  # High as sequences for zero-shot should be shorter due to StoppingCriteria trigger
            tokenizer=dataset.tokenizer,
            remove_user_tokens_from_sequences=self._cfg.model.remove_user_tokens_from_sequences,
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

        non_items_kg_elements = dataset.field2id_token[dataset.entity_field][dataset.item_num :]  # skip items ids
        items_kg_elements = list(dataset.entity2item.keys())
        kg_elements = np.concatenate([items_kg_elements, non_items_kg_elements])
        no_id_kg_elements_map = {}
        for el in kg_elements:
            name = ".".join(el.split(".", maxsplit=2)[::2])  # skips numeric ID
            no_id_kg_elements_map.setdefault(name, []).append(el)

        self._ready = True
        self._logger.info("RecommenderService: Initialization complete.")

    def is_ready(self) -> bool:
        return self._ready

    async def shutdown(self) -> None:
        # TODO: release resources if needed
        self._ready = False
        self._logger and self._logger.info("Core shutdown")

    async def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        #TODO: implement gpu offloading if needed
        if not self._ready:
            raise RuntimeError("Service not ready")

        return dummy_place_recommender(**payload)

    async def fetch_place_info(self, place: str) -> Dict[str, Any]:
        #TODO: implement gpu offloading if needed
        if not self._ready:
            raise RuntimeError("Service not ready")

        with self._neo4j_driver.session() as session:
            info = fetch_place_info(session, place, logger=self._logger)
        return info

    async def search_places(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if not self._ready:
            raise RuntimeError("Service not ready")

        with self._neo4j_driver.session() as session:
            from src.core.info import search
            results = search(
                session,
                query.get("query", ""),
                limit=query.get("limit", 10),
                position=query.get("position"),
                distance=query.get("distance", 1000.0),
                categories=query.get("categories"),
                logger=self._logger,
            )
            
            self._logger.info(f"Fetching info for: {results}")
            results = {"results": [fetch_place_info(session, r["name"], logger=self._logger) for r in results if r["name"] is not None]}
        return results