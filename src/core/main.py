# Dummy function representing your food recommender system
def food_recommender(  # noqa: PLR0913
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

    return ["Spaghetti carbonara", "Fettuccine alfredo", "Pennette with basil pesto"], [0.8, 0.7, 0.6]
