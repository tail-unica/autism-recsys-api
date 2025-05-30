# Dummy function representing the food recommender system
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
