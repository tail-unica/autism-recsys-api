import logging
from typing import Literal, Optional

import polars as pl

from src.core import (
    all_food_semantic_matcher,
    cfg,
    food_data,
    ingredients_only_semantic_matcher,
    recipes_only_semantic_matcher,
)
from src.core.semantic_matcher import HierarchicalSemanticMatcher

logger = logging.getLogger("PHaSE API")


def get_food_info(food_item: str):
    """Retrieves detailed information about a specific food item or recipe from the food database.

    Args:
        food_item (str): Name of the food item to fetch information for.

    Returns:
        dict: Food information including healthiness score, sustainability score, nutritional values, and ingredients.
    """
    info_columns = cfg.core.info_columns
    matched_food_items = food_data.filter(pl.col("name") == food_item).select(info_columns)
    # TODO: matched_food_items actually can have multiple rows due to both ingredients and recipes being repeated
    # Think about how to better handle this
    matched_row = matched_food_items.first()

    food_item_type = matched_row.select("food_item_type").collect().item()

    # Mapping scores to qualitative formats
    qualitative_scores_dict = get_health_sustainability_info(matched_row)

    # If nutritional values or ingredients structs only contain None, set them to None
    nutritional_values = validate_struct(matched_row, "nutritional_values")
    ingredients_dict = validate_struct(matched_row, "flat_ingredients")

    food_item_url = matched_row.select("recipe_url").collect().item()

    return {
        "food_item": food_item,
        "food_item_type": food_item_type,
        "healthiness": qualitative_scores_dict["healthiness"],
        "sustainability": qualitative_scores_dict["sustainability"],
        "nutritional_values": nutritional_values,
        "ingredients": ingredients_dict,
        "food_item_url": food_item_url,
    }


def get_health_sustainability_info(matched_row: pl.LazyFrame) -> dict:
    """Retrieves healthiness and sustainability information from a matched food item row.

    Args:
        matched_row (pl.LazyFrame): LazyFrame containing the matched food item information.

    Returns:
        dict: Dictionary with healthiness and sustainability scores and qualitative descriptions.
    """
    healthiness_score = (
        matched_row.select(pl.col("healthiness_score_groups").struct.field(cfg.core.healthiness_metric))
        .collect()
        .item()
    )
    sustainability = matched_row.select(
        pl.col("environmental_impact").struct.field(["CF", "WF", "sustainability_score_group"])
    ).collect()
    sustainability_score = sustainability["sustainability_score_group"].item()

    health_sust_info = {"healthiness": None, "sustainability": None}
    if healthiness_score is not None:
        health_sust_info["healthiness"] = {
            "score": healthiness_score,
            "qualitative": map_score_to_qualitative(healthiness_score, "healthiness"),
        }
    if sustainability_score is not None:
        health_sust_info["sustainability"] = {
            "score": sustainability_score,
            "qualitative": map_score_to_qualitative(sustainability_score, "sustainability"),
            "CF": sustainability["CF"].item(),
            "WF": sustainability["WF"].item(),
        }

    return health_sust_info


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


def validate_struct(df: pl.LazyFrame, struct_col: str) -> Optional[pl.Series]:
    """Validates a struct to ensure it does not contain only None values.

    Args:
        df (pl.LazyFrame): The Polars LazyFrame containing the struct.
        struct_col (str): The name of the struct column to validate.

    Returns:
        Optional[pl.Series]: The validated struct, or None if it contains only None values.
    """
    if (df.select(pl.col(struct_col).struct.unnest()).select(pl.all_horizontal(pl.all().is_null()))).collect().item():
        return None
    return df.select(struct_col).collect().item()


def get_food_semantic_matcher(
    food_item_type: Optional[Literal["ingredient", "recipe"]] = None,
) -> HierarchicalSemanticMatcher:
    """Returns the appropriate semantic matcher based on the food item type.

    Args:
        food_item_type (str): Type of the food item ("ingredient" or "recipe").

    Returns:
        HierarchicalSemanticMatcher: The semantic matcher for the specified food item type.
    """
    if food_item_type == "ingredient":
        return ingredients_only_semantic_matcher
    elif food_item_type == "recipe":
        return recipes_only_semantic_matcher
    else:
        return all_food_semantic_matcher
