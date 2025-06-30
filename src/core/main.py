import logging
from functools import lru_cache

from src.core import cfg, semantic_matcher
from src.core.alternative import filter_healthy_and_sustainable
from src.core.info import get_food_info

logger = logging.getLogger("PHaSE API")


# Dummy function representing the food recommender system
def dummy_food_recommender(  # noqa: PLR0913
    user_id: int,
    *,
    preferences: list[str] = None,
    soft_restrictions: list[str] = None,
    hard_restrictions: list[str] = None,
    previous_recommendations: list[str] = None,
    meal_time: str = None,
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
        ingredients=[
            [("spaghetti", "100g"), ("guanciale", "50g"), ("egg", "1")],
            [("fettuccine", "100g"), ("cream", "50ml"), ("parmesan", "20g")],
            [("pennette", "100g"), ("basil pesto", "20g"), ("olive oil", "10ml")],
        ],
        healthiness_scores=[0.8, 0.7, 0.6],
        sustainability_scores=[0.5, 0.6, 0.7],
        nutritional_values=[
            {"calories": 500.0, "fiber": 5.0, "sugar": 2.0, "carbs": 60.0, "protein": 15.0, "fat": 10.0},
            {"calories": 600.0, "fiber": 3.0, "sugar": 4.0, "carbs": 70.0, "protein": 12.0, "fat": 20.0},
            {"calories": 550.0, "fiber": 4.0, "sugar": 3.0, "carbs": 65.0, "protein": 14.0, "fat": 15.0},
        ],
    )

    return output


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
def food_info_fetcher(food_item: str) -> dict:
    """Enhanced food info fetcher with semantic matching."""

    max_distance_threshold = cfg.semantic_search.max_distance_threshold

    # Find best match
    best_match_name, match_distance = semantic_matcher.find_most_similar_item(
        query=food_item,
        max_distance=max_distance_threshold,
    )

    if best_match_name is None:
        return None

    info = get_food_info(best_match_name)

    return {
        "food_item": best_match_name,
        "food_item_type": info["food_item_type"],
        "healthiness": info["healthiness"],
        "sustainability": info["sustainability"],
        "nutritional_values": info["nutritional_values"],
        "ingredients": info["ingredients"],
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
def food_alternative(food_item: str, k: int) -> dict:
    """Find alternatives for a given food item based on healthiness and sustainability criteria.

    Args:
        food_item (str): Name of the food item to find alternatives for.
        k (int): Number of alternative food items to return.
    """

    max_distance_threshold = cfg.semantic_search.max_distance_threshold

    matches = semantic_matcher.find_similar_items(
        query=food_item,
        top_k=k + 1,  # +1 to get the matched item itself
        max_distance=max_distance_threshold,
    )
    matches, matches_distances = zip(*matches)
    matched_item, alternatives = matches[0], matches[1:]

    logger.info(f"Retrieving information for matched item: {matched_item}")
    matched_item_info = get_food_info(matched_item)  # Fetch info for the matched item
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
