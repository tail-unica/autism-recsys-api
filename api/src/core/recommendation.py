from functools import lru_cache
from typing import Literal, Optional

import torch
from hopwise.data import Interaction

from src.core.info import fetch_place_info
from src.core.recommendation import (
    prepare_recommender_and_raw_inputs_existing_user,
    prepare_recommender_and_raw_inputs_zero_shot,
    reset_logits_processors,
    unpack_recommendation_sequences_tuples,
)
from src.core.utils import cfg, logger

@lru_cache(maxsize=1024)
def food_recommender(  # noqa: PLR0913
    user_id: str,
    *,
    preferences: tuple[str] = None,
    soft_restrictions: tuple[str] = None,
    hard_restrictions: tuple[str] = None,
    previous_recommendations: tuple[str] = None,
    recommendation_count: int = 5,
    diversity_penalty: float = 0.5,
    restrict_preference_graph: bool = False,
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
        restrict_preference_graph (bool, optional):
            Restrict the KG to only the subgraph derived from combining preferences ego-graph. Defaults to False.
    """
    adjusted_recommendation_count = int(recommendation_count * 2)  # Adjust to ensure enough candidates

    # Ensure even number of beams with ceil rounding
    num_beams = int(adjusted_recommendation_count * 1.5) // 2 * 2 + 2
    kwargs = dict(
        max_length=recommender.token_sequence_length,
        min_length=recommender.token_sequence_length,
        paths_per_user=adjusted_recommendation_count,
        num_beams=num_beams,
        num_beam_groups=max(2, num_beams // 2),
        diversity_penalty=diversity_penalty,
        # stopping_criteria=zero_shot_stop_criteria_list,
        return_dict_in_generate=True,
        output_scores=True,
    )
    logger.info(f"Generating recommendations with parameters: {kwargs}")

    if user_id in dataset.field2id_token[dataset.uid_field]:
        logger.info(f"User {user_id} exists in the dataset, using existing user sequence postprocessor.")
        prepare_recommender_and_raw_inputs_existing_user(
            user_id,
            dataset,
            recommender,
        )
    else:
        logger.info(f"User {user_id} does not exist in the dataset, using zero-shot sequence postprocessor.")
        raw_inputs, matched_previous_recommendations = prepare_recommender_and_raw_inputs_zero_shot(
            dataset,
            preferences=preferences,
            previous_recommendations=previous_recommendations,
            hard_restrictions=hard_restrictions,
            soft_restrictions=soft_restrictions,
            restrict_preference_graph=restrict_preference_graph,
        )

    logger.info("Tokenizing raw inputs for recommendation generation...")
    inputs = dataset.tokenizer(raw_inputs, return_tensors="pt", add_special_tokens=False).to(cfg.recommender.device)
    inputs = Interaction(inputs.data)

    valid_inputs_mask = torch.isin(
        inputs["input_ids"][:, 1:], torch.tensor(dataset.tokenizer.all_special_ids, device=inputs["input_ids"].device)
    ).squeeze(dim=1)
    if valid_inputs_mask.all():
        logger.error("All input tokens are special tokens. Returning None.")
        return None

    inputs = inputs[torch.logical_not(valid_inputs_mask)]
    # TODO: the number of final sequences depend on the number of valid inputs, not on the recommendation count
    # so for each sequence we should generate recommendation_count / len(inputs) recommendations
    # with a minimum of 1 for each input

    logger.info(f"Executing generation with inputs: {raw_inputs}")
    try:
        outputs = recommender.generate(inputs, **kwargs)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return None

    logger.info("Processing outputs to get recommendations...")
    max_new_tokens = recommender.token_sequence_length - inputs["input_ids"].size(1)
    _, sequences = recommender.sequence_postprocessor.get_sequences(
        outputs, max_new_tokens=max_new_tokens, previous_recommendations=matched_previous_recommendations
    )

    # for seq in sequences:
    #     seq[-1] = recommender.decode_path(seq[-1])

    top_rec_index = sorted(range(len(sequences)), key=lambda i: sequences[i][2], reverse=True)[:recommendation_count]
    sequences = [sequences[i] for i in top_rec_index]
    unpacked_sequences = unpack_recommendation_sequences_tuples(sequences, dataset, user_id)
    if unpacked_sequences is None:
        return None
    else:
        scores, recommendations, explanations = unpacked_sequences

    try:
        recommendations_info = [food_info_fetcher(rec, food_item_type="recipe") for rec in recommendations]
    except Exception as e:
        logger.error(f"Error fetching food info: {e}")
        return None

    reset_logits_processors(recommender.logits_processor_list)

    return dict(
        recommendations=recommendations,
        scores=scores,
        explanations=explanations,
        recommendations_info=recommendations_info,
    )
