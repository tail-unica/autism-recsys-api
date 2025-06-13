import logging
from typing import Optional

import polars as pl

from src.core import cfg, food_data

logger = logging.getLogger("PHaSE API")


def get_food_info(food_item: str):
    """Retrieves detailed information about a specific food item or recipe from the food database.

    Args:
        food_item (str): Name of the food item to fetch information for.

    Returns:
        dict: Food information including healthiness score, sustainability score, nutritional values, and ingredients.
    """
    info_columns = [
        "name",
        "food_item_type",
        "healthiness_score_groups",
        "environmental_impact",
        "nutritional_values",
        "flat_ingredients",
    ]
    matched_food_items = food_data.filter(pl.col("name") == food_item).select(info_columns).collect()
    # TODO: matched_row actually can have multiple rows due to both ingredients and recipes being repeated
    # Think about how to better handle this

    if matched_food_items.height > 1:
        matched_row = matched_food_items[0]
    else:
        matched_row = matched_food_items

    food_item_type = matched_row["food_item_type"].item()

    # Mapping scores to qualitative formats
    qualitative_scores_dict = map_scores_to_qualitative(matched_row)

    # If nutritional values or ingredients structs only contain None, set them to None
    nutritional_values = validate_series_struct(matched_row["nutritional_values"])
    ingredients_dict = validate_series_struct(matched_row["flat_ingredients"])

    return {
        "food_item": food_item,
        "food_item_type": food_item_type,
        "healthiness_score": qualitative_scores_dict["healthiness"]["score"],
        "qualitative_healthiness": qualitative_scores_dict["healthiness"]["qualitative"],
        "sustainability_score": qualitative_scores_dict["sustainability"]["score"],
        "qualitative_sustainability": qualitative_scores_dict["sustainability"]["qualitative"],
        "nutritional_values": nutritional_values,
        "ingredients": ingredients_dict,
    }


def map_scores_to_qualitative(matched_row: pl.DataFrame) -> dict:
    """Maps healthiness and sustainability scores to qualitative descriptions.

    Args:
        matched_row (pl.DataFrame): DataFrame containing the matched food item information.

    Returns:
        dict: Dictionary with qualitative descriptions for healthiness and sustainability scores.
    """
    healthiness_score = matched_row["healthiness_score_groups"].item()[cfg.core.healthiness_metric]
    sustainability_score = matched_row["environmental_impact"].item()["sustainability_score_group"]

    return {
        "healthiness": {
            "score": healthiness_score,
            "qualitative": map_score_to_qualitative(healthiness_score, "healthiness"),
        },
        "sustainability": {
            "score": sustainability_score,
            "qualitative": map_score_to_qualitative(sustainability_score, "sustainability"),
        },
    }


def map_score_to_qualitative(score: str, score_type: str) -> str:
    """Maps a score to a qualitative description based on the score type (healthiness or sustainability).

    Args:
        score (str): The score to map (e.g., "A", "B", "C", "D", "E").
        score_type (str): The type of score ("healthiness" or "sustainability").

    Returns:
        str: Qualitative description of the score.
    """
    score_map = cfg.core.score_map
    if score is not None and score in score_map:
        qualitative_score = f"{score_map[score]} {score_type} level"
    else:
        logger.warning(f"Food item has no {score_type} score, or {score} is not in the score map")
        qualitative_score = f"Unavailable {score_type} level"

    return qualitative_score


def validate_series_struct(series: pl.DataFrame) -> Optional[pl.Series]:
    """Validates a struct to ensure it does not contain only None values.

    Args:
        series (pl.Series): The Polars Series containing the struct to validate.

    Returns:
        Optional[pl.Series]: The validated struct, or None if it contains only None values.
    """
    if series.struct.unnest().select(pl.any_horizontal(pl.all().is_null())).item() is None:
        return None
    return series.item()
