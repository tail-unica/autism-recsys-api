from __future__ import annotations
import asyncio
from typing import Any, Dict, Optional
from neo4j import GraphDatabase
import os
import numpy as np

from src.core.utils import get_cfg
from src.core.logger import logger
from src.core.dummy import dummy_place_info_fetcher, dummy_place_recommender
from src.core.info import fetch_place_info
from src.core.data import load_data
from src.core.recommendation import (
    prepare_recommender_and_raw_inputs_existing_user,
    prepare_recommender_and_raw_inputs_zero_shot,
    unpack_recommendation_sequences_tuples,
    reset_logits_processors,
)
from src.core.recommendation_tools import (
    RestrictionLogitsProcessorWordLevel,
    ZeroShotConstrainedLogitsProcessor,
    ZeroShotCumulativeSequenceScorePostProcessor,
)

import torch
from hopwise.utils import get_model, init_seed
from hopwise.data.utils import PathLanguageModelingTokenType, create_dataset, data_preparation
from hopwise.model.logits_processor import LogitsProcessorList
from hopwise.model.sequence_postprocessor import CumulativeSequenceScorePostProcessor
from hopwise.data import Interaction
from safetensors.torch import load_file
from transformers import AutoTokenizer


class RecommenderService:
    def __init__(self) -> None:
        self.cfg = None
        self._ready = False
        self._warmup_task: Optional[asyncio.Task] = None

        self._neo4j_driver = None

        self.recommender = None
        self.dataset = None

        self.constrained_logits_processors_list = None
        self.existing_user_cumulative_sequence_postprocessor = None
        self.no_id_kg_elements_map = None
        self.zero_shot_constrained_logits_processors_list = None
        self.zero_shot_sequence_postprocessor = None

    @property
    def logger(self):
        return logger

    async def initialize(self) -> None:
        """Initialize the recommender service with Neo4j database, model, and processors."""
        self.cfg = get_cfg()

        logger.info("RecommenderService: Starting initialization.")

        # ===== Neo4j Database Setup =====
        self._neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )
        logger.info("Connected to Neo4j database.")

        # ===== Load and prepare dataset =====
        logger.info("Loading dataset...")
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        with self._neo4j_driver.session() as session:
            load_data(session, data_dir, dataset=self.cfg.data.dataset)

        # ===== Load model checkpoint and configuration =====
        logger.info("Loading checkpoint...")
        checkpoint = torch.load(
            self.cfg.model.hopwise_checkpoint_file, 
            map_location=self.cfg.model.device, 
            weights_only=False,
        )
        config = checkpoint["config"]
        config["checkpoint_dir"] = os.path.dirname(self.cfg.model.hopwise_checkpoint_file)
        config["data_path"] = os.path.join(data_dir, self.cfg.data.dataset)
        config["load_col"]["item"] = ["poi_id", "name"]
        config._set_env_behavior()

        # ===== Initialize dataset and data splits =====
        logger.info("Initializing dataset...")
        init_seed(config["seed"], config["reproducibility"])
        self.dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, self.dataset)

        # ===== Create recommender model and load weights =====
        logger.info("Initializing recommender and applying checkpoint weights...")
        self.recommender = get_model(config["model"])(config, train_data.dataset)
        self.recommender = self.recommender.to(device=self.cfg.model.device, dtype=config["weight_precision"])

        # ===== Load pretrained weights from HuggingFace checkpoint =====
        hf_checkpoint_file = self.cfg.model.hopwise_checkpoint_file.replace("hopwise", "huggingface")
        weights = load_file(os.path.join(hf_checkpoint_file, "model.safetensors"))
        self.recommender.load_state_dict(weights, strict=False)
        self.dataset._tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint_file)

        # ===== Model Compilation and Post-processing Setup =====
        logger.info("Compiling recommender for performance...")
        self.recommender = torch.compile(self.recommender, mode=self.cfg.model.compile_mode)

        # ===== Initialize post-processors for sequence scoring =====
        logger.info("Creating post-processors and logits processors for recommendation generation...")
        self.existing_user_cumulative_sequence_postprocessor = CumulativeSequenceScorePostProcessor(
            self.dataset.tokenizer, self.dataset.get_user_used_ids(), self.dataset.item_num
        )
        self.zero_shot_sequence_postprocessor = ZeroShotCumulativeSequenceScorePostProcessor(self.dataset.tokenizer, self.dataset.item_num)

        # ===== Logits Processors Setup =====
        self.constrained_logits_processors_list = self.recommender.logits_processor_list

        # ===== Setup zero-shot logits processors for constrained generation =====
        zero_shot_restriction_logits_processor = RestrictionLogitsProcessorWordLevel(
            tokenized_ckg=self.dataset.get_tokenized_ckg(),
            tokenizer=self.dataset.tokenizer,
            propagate_connected_entities=self.cfg.model.propagate_connected_entities,
        )

        ui_relation = self.dataset.field2token_id[self.dataset.relation_field][self.dataset.ui_relation]
        zero_shot_constrained_logits_processor = ZeroShotConstrainedLogitsProcessor(
            tokenized_ckg=self.dataset.get_tokenized_ckg(),
            tokenized_used_ids=self.dataset.get_tokenized_used_ids(),
            max_sequence_length=10,  # High as sequences for zero-shot should be shorter due to StoppingCriteria trigger
            tokenizer=self.dataset.tokenizer,
            remove_user_tokens_from_sequences=self.cfg.model.remove_user_tokens_from_sequences,
            tokenized_ui_relation=(
                self.dataset.tokenizer.convert_tokens_to_ids(PathLanguageModelingTokenType.RELATION.token + str(ui_relation))
            ),
        )
        self.zero_shot_constrained_logits_processors_list = LogitsProcessorList(
            [
                zero_shot_constrained_logits_processor,
                zero_shot_restriction_logits_processor,
            ]
        )

        # ===== Knowledge Graph Elements Mapping =====
        # Build mapping of knowledge graph elements without numeric IDs
        non_items_kg_elements = self.dataset.field2id_token[self.dataset.entity_field][self.dataset.item_num :]  # skip items ids
        items_kg_elements = list(self.dataset.entity2item.keys())
        kg_elements = np.concatenate([items_kg_elements, non_items_kg_elements])
        self.no_id_kg_elements_map = {}
        for el in kg_elements:
            name = ".".join(el.split(".", maxsplit=2)[::2])  # skips numeric ID
            self.no_id_kg_elements_map.setdefault(name, []).append(el)

        # ===== Mark service as ready =====
        self._ready = True
        logger.info("RecommenderService: Initialization complete.")

    def is_ready(self) -> bool:
        return self._ready

    async def shutdown(self) -> None:
        # TODO: release resources if needed
        self._ready = False
        logger.info("Core shutdown")


    async def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend places based on user input.

        Args:
            payload: Dictionary containing recommendation parameters.
                * user_id: str - Unique identifier for the user.
                * preferences: Optional[List[str]] - List of user preferences.
                * previous_recommendations: Optional[List[str]] - List of previously recommended places.
                * recommendation_count: Optional[int] - Number of recommendations to return.
                * diversity_factor: Optional[float] - Diversity factor for recommendations.
                * restrict_preferences: Optional[bool] - Whether to restrict recommendations based on user preferences.
                * aversions: List[Dict[str, float]] - Sensory idiosyncratic aversion details for the user.
        Returns:
            Dictionary with recommended places and their information.
        Raises:
            RuntimeError: If the service is not ready.
        """
        # Validate service readiness
        if not self._ready:
            raise RuntimeError("Service not ready")

        # ===== Extract parameters from payload =====
        user_id = payload.get("user_id")
        preferences = payload.get("preferences")
        previous_recommendations = payload.get("previous_recommendations", [])
        recommendation_count = payload.get("recommendation_count", 5)
        diversity_factor = payload.get("diversity_factor", 0.5)
        hard_restrictions = payload.get("hard_restrictions", None)
        soft_restrictions = payload.get("soft_restrictions", None)
        restrict_preference_graph = payload.get("restrict_preferences", False)
        

        # ===== Prepare generation parameters =====
        adjusted_recommendation_count = int(recommendation_count * 2)
        num_beams = int(adjusted_recommendation_count * 1.5) // 2 * 2 + 2

        kwargs = dict(
            max_length=self.recommender.token_sequence_length,
            min_length=self.recommender.token_sequence_length,
            paths_per_user=adjusted_recommendation_count,
            num_beams=num_beams,
            num_beam_groups=max(2, num_beams // 2),
            diversity_penalty=diversity_factor,
            return_dict_in_generate=True,
            output_scores=True,
        )

        logger.info(f"Generating recommendations with parameters: {kwargs}")

        # ===== Prepare raw inputs based on user existence =====
        if user_id in self.dataset.field2id_token[self.dataset.uid_field]:
            logger.info(f"User {user_id} exists in dataset, using existing user sequence postprocessor.")
            raw_inputs = prepare_recommender_and_raw_inputs_existing_user(
                self.recommender,
                self.dataset,
                self.existing_user_cumulative_sequence_postprocessor,
                self.constrained_logits_processors_list,
                user_id,
            )
        else:
            logger.info(f"User {user_id} does not exist in dataset, using zero-shot sequence postprocessor.")
            raw_inputs = prepare_recommender_and_raw_inputs_zero_shot(
                self.recommender,
                self.dataset,
                self.zero_shot_sequence_postprocessor,
                self.zero_shot_constrained_logits_processors_list,
                preferences=preferences,
                previous_recommendations=previous_recommendations,
                hard_restrictions=hard_restrictions,
                soft_restrictions=soft_restrictions,
                restrict_preference_graph=restrict_preference_graph,
            )

        # ===== Tokenize inputs =====
        logger.info("Tokenizing raw inputs for recommendation generation...")
        inputs = self.dataset.tokenizer(raw_inputs, return_tensors="pt", add_special_tokens=False).to(self.cfg.model.device)
        inputs = Interaction(inputs.data)

        # ===== Validate inputs =====
        valid_inputs_mask = torch.isin(
            inputs["input_ids"][:, 1:], torch.tensor(self.dataset.tokenizer.all_special_ids, device=inputs["input_ids"].device)
        ).squeeze(dim=1)
        if valid_inputs_mask.all():
            logger.error("All input tokens are special tokens. Returning None.")
            return None

        inputs = inputs[torch.logical_not(valid_inputs_mask)]

        # ===== Generate recommendations =====
        logger.info(f"Executing generation with {inputs['input_ids'].shape[0]} input samples")
        try:
            outputs = self.recommender.generate(inputs, **kwargs)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None

        # ===== Process outputs =====
        logger.info("Processing outputs to get recommendations...")
        max_new_tokens = self.recommender.token_sequence_length - inputs["input_ids"].size(1)
        _, sequences = self.recommender.sequence_postprocessor.get_sequences(
            outputs, max_new_tokens=max_new_tokens, previous_recommendations=previous_recommendations
        )

        # ===== Select top recommendations =====
        top_rec_index = sorted(range(len(sequences)), key=lambda i: sequences[i][2], reverse=True)[:recommendation_count]
        sequences = [sequences[i] for i in top_rec_index]
        unpacked_sequences = self.unpack_recommendation_sequences(sequences, user_id)
        if unpacked_sequences is None:
            return None

        scores, recommendations, explanations = unpacked_sequences

        # ===== Fetch place information =====
        try:
            recommendations_info = [await self.fetch_place_info(rec) for rec in recommendations]
        except Exception as e:
            logger.error(f"Error fetching place info: {e}")
            return None

        # ===== Reset logits processors =====
        reset_logits_processors(self.recommender.logits_processor_list)

        # ===== Build recommendation items with schema =====
        recommendation_items = [
            {
                "place": recommendations[i],
                "score": scores[i],
                "explanation": explanations[i],
                "metadata": recommendations_info[i],
            }
            for i in range(len(recommendations))
        ]

        return dict(
            user_id=user_id,
            recommendations=recommendation_items,
            conversation_id=None,
        )

    async def fetch_place_info(self, place: str) -> Dict[str, Any]:
        """Fetch detailed information about a place from the Neo4j database.
        Args:
            place: Name of the place to fetch information for.
        Returns:
            Dictionary containing place information.
        Raises:
            RuntimeError: If the service is not ready.
        """
        #TODO: implement gpu offloading if needed
        if not self._ready:
            raise RuntimeError("Service not ready")

        with self._neo4j_driver.session() as session:
            info = fetch_place_info(session, place)
        return info

    async def search_places(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Search for places based on query parameters.
        Args:
            query: Dictionary containing search parameters.
                * query: str - Search keyword.
                * limit: Optional[int] - Maximum number of results to return.
                * position: Optional[Dict[str, float]] - User's geographical position with 'latitude' and 'longitude'.
                * distance: Optional[float] - Search radius in meters.
                * categories: Optional[List[str]] - List of category IDs to filter the search results.
        Returns:
            Dictionary with search results and their information.
        Raises:
            RuntimeError: If the service is not ready.
        """
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
            )
            
            logger.info(f"Fetching info for: {results}")
            results = {"results": [fetch_place_info(session, r["name"]) for r in results if r["name"] is not None]}
        return results