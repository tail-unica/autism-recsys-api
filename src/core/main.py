import logging
from functools import lru_cache
from typing import Literal, Optional

import torch
from hopwise.data import Interaction
from hopwise.utils import PathLanguageModelingTokenType

from src.core import (
    cfg,
    constrained_logits_processors_list,
    dataset,
    existing_user_cumulative_sequence_postprocessor,
    kg_elements_semantic_matcher,
    recommender,
    zero_shot_constrained_logits_processors_list,
    zero_shot_sequence_postprocessor,
    # zero_shot_stop_criteria_list,
)
from src.core.alternative import filter_healthy_and_sustainable
from src.core.info import get_food_info, get_food_semantic_matcher
from src.core.recommendation import (
    RestrictionLogitsProcessorWordLevel,
    prepare_zero_shot_raw_inputs,
    set_restrictions,
    token2real_token,
)

logger = logging.getLogger("PHaSE API")


# Dummy function representing the food recommender system
def dummy_food_recommender(  # noqa: PLR0913
    user_id: int,
    *,
    preferences: list[str] = None,
    soft_restrictions: list[str] = None,
    hard_restrictions: list[str] = None,
    previous_recommendations: list[str] = None,
    recommendation_count: int = 5,
    diversity_factor: float = 0.5,
) -> list[str]:
    """Dummy food recommender system.

    Args:
        user_id (int): Unique identifier for the user
        preferences (list[str], optional):
            List of food items, ingredients, or cuisines the user likes. Defaults to None.
        soft_restrictions (list[str], optional):
        List of food items, ingredients, or cuisines the user dislikes. Defaults to None.
        hard_restrictions (list[str], optional):
            List of specific food items to completely exclude from recommendations. Defaults to None.
        previous_recommendations (list[str], optional):
            List of previously recommended items to avoid repetition. Defaults to None.
        meal_time (str, optional):
            What meal the user is looking for (breakfast, lunch, dinner, snack). Defaults to None.
        recommendation_count (int, optional): Number of recommendations to return. Defaults to 5.
        diversity_factor (float, optional):
            Controls how diverse the recommendations should be (0.0-1.0). Defaults to 0.5.
    """

    output = dict(
        recommendations=[
            "Spaghetti carbonara",
            "Fettuccine alfredo",
            "Pennette with basil pesto",
        ],
        scores=[0.8, 0.7, 0.6],
        explanations=[
            "U25 interacted_with 'Pasta amatriciana' has_ingredient 'guanciale' has_ingredient 'Spaghetti Carbonara'",
            "U25 interacted_with 'Pasta broccoli' has_indicator 'HIGH PROTEIN' has_indicator 'Fettuccine Alfredo'",
            "U25 interacted_with 'Pizza genovese' has_tag 'pesto' has_tag 'Pennette with basil pesto'",
        ],
        recommendations_info=[
            {
                "food_item": "Spaghetti carbonara",
                "food_item_type": "recipe",
                "ingredients_dict": {
                    "ingredients": ["spaghetti", "guanciale", "egg"],
                    "quantities": ["100g", "50g", "1"],
                },
                "healthiness": {"score": "B", "qualitative": "Good healthiness level"},
                "sustainability": {"score": "C", "qualitative": "Fair sustainability level"},
                "nutritional_values": {
                    "calories [cal]": 500,
                    "protein [g]": 15,
                    "carbohydrates [g]": 60,
                    "fats [g]": 10,
                },
                "food_item_url": "https://www.example.com/spaghetti-carbonara",
            },
            {
                "food_item": "Fettuccine alfredo",
                "food_item_type": "recipe",
                "ingredients_dict": {
                    "ingredients": ["fettuccine", "cream", "parmesan"],
                    "quantities": ["100g", "50ml", "20g"],
                },
                "healthiness": {"score": "C", "qualitative": "Moderate healthiness level"},
                "sustainability": {"score": "B", "qualitative": "Good sustainability level"},
                "nutritional_values": {
                    "calories [cal]": 600,
                    "protein [g]": 12,
                    "carbohydrates [g]": 70,
                    "fats [g]": 20,
                },
                "food_item_url": "https://www.example.com/fettuccine-alfredo",
            },
            {
                "food_item": "Pennette with basil pesto",
                "food_item_type": "recipe",
                "ingredients_dict": {
                    "ingredients": ["pennette", "basil pesto", "olive oil"],
                    "quantities": ["100g", "20g", "10ml"],
                },
                "healthiness": None,
                "sustainability": None,
                "nutritional_values": {
                    "calories [cal]": 550,
                    "protein [g]": 14,
                    "carbohydrates [g]": 65,
                    "fats [g]": 15,
                },
                "food_item_url": "https://www.example.com/pennette-basil-pesto",
            },
        ],
    )

    return output


@lru_cache(maxsize=1024)
def food_recommender(  # noqa: PLR0913, PLR0915
    user_id: str,
    *,
    preferences: tuple[str] = None,
    soft_restrictions: tuple[str] = None,
    hard_restrictions: tuple[str] = None,
    previous_recommendations: tuple[str] = None,
    recommendation_count: int = 5,
    diversity_penalty: float = 0.5,
) -> dict:
    """Food recommender system.

    Args:
        user_id (int): Unique identifier for the user
        preferences (list[str], optional):
            List of food items, ingredients, or cuisines the user likes. Defaults to None.
        soft_restrictions (list[str], optional):
            List of food items, ingredients, or cuisines the user dislikes. Defaults to None.
        hard_restrictions (list[str], optional):
            List of specific food items to completely exclude from recommendations. Defaults to None.
        previous_recommendations (list[str], optional):
            List of previously recommended items to avoid repetition. Defaults to None.
        recommendation_count (int, optional): Number of recommendations to return. Defaults to 5.
        diversity_penalty (float, optional):
            Controls how diverse the recommendations should be (0.0-1.0). Defaults to 0.5.
    """
    num_beams = int(recommendation_count * 1.5) // 2 * 2 + 2  # Ensure even number of beams with ceil rounding
    kwargs = dict(
        max_length=recommender.token_sequence_length,
        min_length=recommender.token_sequence_length,
        paths_per_user=recommendation_count,
        num_beams=num_beams,
        num_beam_groups=num_beams // 2,
        diversity_penalty=diversity_penalty,
        # stopping_criteria=zero_shot_stop_criteria_list,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logger.info(f"Generating recommendations with parameters: {kwargs}")

    if user_id in dataset.field2id_token[dataset.uid_field]:
        logger.info(f"User {user_id} exists in the dataset, using existing user sequence postprocessor.")
        logger.info("Preparing raw inputs for existing user...")
        ui_relation = dataset.field2token_id[dataset.relation_field][dataset.ui_relation]
        raw_inputs = [
            dataset.path_token_separator.join(
                [
                    dataset.tokenizer.bos_token,
                    PathLanguageModelingTokenType.USER.token + user_id,
                    PathLanguageModelingTokenType.RELATION.token + str(ui_relation),
                ]
            )
        ]
        recommender.sequence_postprocessor = existing_user_cumulative_sequence_postprocessor
        recommender.logits_processor_list = constrained_logits_processors_list
    else:
        logger.info(f"User {user_id} does not exist in the dataset, using zero-shot sequence postprocessor.")
        logger.info("Finding best matches for preferences and preparing raw inputs...")
        raw_inputs = prepare_zero_shot_raw_inputs(preferences, dataset, kg_elements_semantic_matcher)
        recommender.sequence_postprocessor = zero_shot_sequence_postprocessor
        recommender.logits_processor_list = zero_shot_constrained_logits_processors_list

        # if previous_recommendations:
        #     breakpoint()
        #     previous_recommendations = []

        zero_shot_constrained_logits_processors_list[0].previous_recommendations = previous_recommendations

        if any(
            isinstance(logit_processor, RestrictionLogitsProcessorWordLevel)
            for logit_processor in recommender.logits_processor_list
        ):
            logger.info("Setting restrictions if any...")
            if hard_restrictions or soft_restrictions:
                set_restrictions(
                    hard_restrictions=hard_restrictions,
                    soft_restrictions=soft_restrictions,
                    kg_elements_semantic_matcher=kg_elements_semantic_matcher,
                    zero_shot_constrained_logits_processors_list=zero_shot_constrained_logits_processors_list,
                    logger=logger,
                )

    logger.info("Tokenizing raw inputs for recommendation generation...")
    inputs = dataset.tokenizer(raw_inputs, return_tensors="pt", add_special_tokens=False).to(cfg.recommender.device)
    inputs = Interaction(inputs.data)

    valid_inputs_mask = torch.isin(
        inputs["input_ids"][:, 1:], torch.tensor(dataset.tokenizer.all_special_ids, device=inputs["input_ids"].device)
    ).squeeze()
    if valid_inputs_mask.all():
        logger.error("All input tokens are special tokens. Returning None.")
        return None

    inputs = inputs[torch.logical_not(valid_inputs_mask)]

    logger.info(f"Executing generation with inputs: {raw_inputs}")
    try:
        outputs = recommender.generate(inputs, **kwargs)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return None

    logger.info("Processing outputs to get recommendations...")
    max_new_tokens = recommender.token_sequence_length - inputs["input_ids"].size(1)
    scores, sequences = recommender.sequence_postprocessor.get_sequences(
        outputs, max_new_tokens=max_new_tokens, previous_recommendations=previous_recommendations
    )

    # for seq in sequences:
    #     seq[-1] = recommender.decode_path(seq[-1])

    recommendation_ids = [seq[1] for seq in sequences]
    scores = [seq[2] for seq in sequences]
    explanations = [seq[3] for seq in sequences]
    for idx in range(len(explanations)):
        explanations[idx] = [token2real_token(token, dataset) for token in explanations[idx][1:]]

    try:
        if user_id not in dataset.field2id_token[dataset.uid_field]:
            explanations = [
                f"User {user_id} has_preference " + " ".join(exp).replace(dataset.ui_relation, "interacted_with")
                for exp in explanations
            ]

        mapped_recommendations = dataset.field2id_token["name"][dataset.item_feat[recommendation_ids]["name"]].tolist()
        recommendations = [" ".join(filter(lambda x: x != "[PAD]", x)) for x in mapped_recommendations]

        recommendations_info = [food_info_fetcher(rec, food_item_type="recipe") for rec in recommendations]

        if any(
            isinstance(logit_processor, RestrictionLogitsProcessorWordLevel)
            for logit_processor in recommender.logits_processor_list
        ):
            # Clear restrictions after generation
            zero_shot_constrained_logits_processors_list[-1].clear_restrictions()
    except Exception as e:
        logger.error(f"Error processing recommendations: {e}")
        return None

    zero_shot_constrained_logits_processors_list[0].previous_recommendations = None

    return dict(
        recommendations=recommendations,
        scores=scores,
        explanations=explanations,
        recommendations_info=recommendations_info,
    )


# Dummy function representing the food info fetcher
def dummy_food_info_fetcher(food_item: str) -> dict:
    """Dummy food info fetcher.

    Args:
        food_item (str): Name of the food item to get information about.
    """
    return {
        "food_item": food_item,
        "food_item_type": "recipe",
        "healthiness": {
            "score": "B",
            "qualitative": "Good healthiness level",
        },
        "sustainability": {
            "score": "C",
            "qualitative": "Fair sustainability level",
            "CF": 0.8,
            "WF": 0.7,
        },
        "nutritional_values": {
            "calories [cal]": 100,
            "protein [g]": 5,
            "carbohydrates [g]": 20,
            "fats [g]": 2,
        },
        "ingredients": {"ingredients": ["ingredient1", "ingredient2"], "quantities": ["100g", "50g"]},
        "food_item_url": "https://www.example.com/recipe",
    }


@lru_cache(maxsize=1024)
def food_info_fetcher(food_item: str, food_item_type: Optional[Literal["ingredient", "recipe"]] = None) -> dict:
    """Enhanced food info fetcher with semantic matching."""

    max_distance_threshold = cfg.semantic_search.max_distance_threshold

    # Find best match
    best_match = get_food_semantic_matcher(food_item_type=food_item_type).find_most_similar_item(
        query=food_item,
        max_distance=max_distance_threshold,
    )

    if not best_match:
        logger.warning(f"No match found for food item: {food_item}")
        return None

    best_match_name, _ = best_match
    best_match_info = get_food_info(best_match_name)

    return {
        "food_item": best_match_name,
        "food_item_type": best_match_info["food_item_type"],
        "healthiness": best_match_info["healthiness"],
        "sustainability": best_match_info["sustainability"],
        "nutritional_values": best_match_info["nutritional_values"],
        "ingredients": best_match_info["ingredients"],
    }


def dummy_food_alternative(food_item: str, k: int) -> dict:
    """Dummy food alternative recommender.

    Args:
        food_item (str): Name of the food item to find alternatives for.
        k (int): Number of alternative food items to return.
    """
    return {
        "matched_food_item": dummy_food_info_fetcher(food_item),
        "alternatives": [
            dummy_food_info_fetcher("Pasta alla gricia"),
            dummy_food_info_fetcher("Fettuccine Alfredo"),
            dummy_food_info_fetcher("Penne with basil pesto"),
        ],
    }


@lru_cache(maxsize=1024)
def food_alternative(food_item: str, k: int, food_item_type: Optional[Literal["ingredient", "recipe"]] = None) -> dict:
    """Find alternatives for a given food item based on healthiness and sustainability criteria.

    Args:
        food_item (str): Name of the food item to find alternatives for.
        k (int): Number of alternative food items to return.
    """

    max_distance_threshold = cfg.semantic_search.max_distance_threshold

    logger.info(f"Finding best match of {food_item} and retrieving its info")
    matched_item_info = food_info_fetcher(food_item, food_item_type=food_item_type)
    matches = get_food_semantic_matcher(food_item_type=food_item_type).find_similar_items(
        query=matched_item_info["food_item"],
        top_k=k + 1,  # +1 to get the matched item itself
        max_distance=max_distance_threshold,
    )
    matches, matches_distances = zip(*matches)
    alternatives = matches[1:]  # Exclude the matched item itself

    logger.info(f"Retrieving information for alternatives: {alternatives}")
    alternatives_info = [get_food_info(alt) for alt in alternatives]
    same_type_alternatives_info_distances = [
        (alt, dist)
        for alt, dist in zip(alternatives_info, matches_distances[1:])
        if alt["food_item_type"] == matched_item_info["food_item_type"]
    ]
    if not same_type_alternatives_info_distances:
        logger.warning(
            f"No alternatives found for {matched_item_info['food_item']} of type {matched_item_info['food_item_type']}"
        )
        return None

    if (healthiness := matched_item_info.get("healthiness", None)) is not None:
        healthiness = healthiness.get("score", None)

    if (sustainability := matched_item_info.get("sustainability", None)) is not None:
        sustainability = sustainability.get("score", None)

    filtered_alternatives = filter_healthy_and_sustainable(
        same_type_alternatives_info_distances,
        healthiness=healthiness,
        sustainability=sustainability,
        distance_weight=cfg.core.alternative_distance_weight,
    )

    if filtered_alternatives is None:
        logger.warning(
            f"No alternatives found for {matched_item_info['food_item']} "
            "that meet healthiness and sustainability criteria"
        )
        return None

    return {
        "matched_food_item": matched_item_info,
        "alternatives": filtered_alternatives["alternatives"],
        "scores": filtered_alternatives["scores"],
    }
