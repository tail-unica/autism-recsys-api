from fastapi import APIRouter, HTTPException

import src.core.main as core
from src.app.schema import (
    AlternativeResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)

logger = core.logger
router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """Food recommendation endpoint.

    **user_id**: Unique identifier for the user

    **preferences**: List of food items, ingredients, or cuisines the user likes
    (some might be dropped if retrieval of filtered candidate items is unfeasible)

    **soft_restrictions**: List of food items, ingredients, or cuisines the user dislikes
    (some might be dropped if retrieval of filtered candidate items is unfeasible)

    **hard_restrictions**: List of specific food items to completely exclude from recommendations

    **meal_time**: What meal the user is looking for (breakfast, lunch, dinner, snack)

    **previous_recommendations**: List of previously recommended items to avoid repetition

    **recommendation_count**: Number of recommendations to return

    **diversity_factor**: Controls how diverse the recommendations should be (0.0-1.0)

    **conversation_id**: Identifier for the conversation these recommendations are associated with
    """
    if hasattr(core, "food_recommender"):
        food_recommender = core.food_recommender
    else:
        logger.warning(
            "Using dummy food recommender. Please ensure the core module is properly configured with a recommender."
        )
        food_recommender = core.dummy_food_recommender

    recommender_output = food_recommender(
        request.user_id,
        preferences=request.preferences,
        soft_restrictions=request.soft_restrictions,
        hard_restrictions=request.hard_restrictions,
        previous_recommendations=request.previous_recommendations,
        meal_time=request.meal_time,
        recommendation_count=request.recommendation_count,
        diversity_factor=request.diversity_factor,
    )

    if recommender_output is None:
        raise HTTPException(
            status_code=404, detail=f"No recommendations found for user {request.user_id} with the specified criteria"
        )

    response = RecommendationResponse(
        user_id=request.user_id,
        recommendations=[
            RecommendationItem(
                name=rec,
                score=score,
                explanation=exp,
                ingredients=ingredients,
                healthiness_score=health_score,
                sustainability_score=sustain_score,
                nutritional_values=nutri_values,
            )
            for rec, score, exp, ingredients, health_score, sustain_score, nutri_values in zip(
                recommender_output["recommendations"],
                recommender_output["scores"],
                recommender_output["explanations"],
                recommender_output["ingredients"],
                recommender_output["healthiness_scores"],
                recommender_output["sustainability_scores"],
                recommender_output["nutritional_values"],
            )
        ],
        conversation_id=request.conversation_id,
    )

    return response


"""
    healthiness = healthiness - healthiness_reference
    sustainability = sustainability - sustainability_reference
    # Negative scores indicate better healthiness and sustainability trade-off
    if healthiness + sustainability > 0:
        return None
    healthiness, sustainability = abs(healthiness), abs(sustainability)

    return (similarity * similarity_weight + healthiness + sustainability) / 3.0
"""


@router.post("/alternative", response_model=AlternativeResponse)
def get_alternative(food_item: str, num_alternatives: int = 5):
    """Alternative food (recipe or ingredient) endpoint. Information about food item is retrieved
    to find alternatives that meet healthiness and sustainability criteria.

    Alternatives are retrieved based on similarity to the provided food item, first according to the food item type
    (recipe or ingredient) and finally filtered to ensure they meet better healthiness and sustainability criteria.
    Specifically, this filtering is performed according to the following procedure:
    1. Retrieve alternatives based on similarity to the provided food item.
    2. For each alternative, retrieve its healthiness and sustainability categorical scores if available.
    3. Map the categorical scores to numerical values, e.g., 'A' -> 1.0, 'B' -> 2.0, etc.
    4. Calculate the score difference between the alternative and the provided food item.
        1. healthiness = healthiness_{alternative} - healthiness_{provided_food_item}
        2. sustainability = sustainability_{alternative} - sustainability_{provided_food_item}
    5. If the sum of healthiness and sustainability is positive, discard the alternative.
    6. Calculate an overall score accounting also for similarity to the provided food item
        > (similarity * similarity_weight + healthiness + sustainability) / 3.0
        where similarity_weight is a hyperparameter that can be tuned in config.
    7. Sort the alternatives by this score in ascending order (lower is better).
    If no alternatives meet the criteria, a 404 error is raised.

    **food_item**: Name of the food item (recipe or ingredient) to get alternatives for
    **num_alternatives**: Number of alternatives to return (default is 5)
    """
    if hasattr(core, "food_alternative"):
        food_alternative = core.food_alternative
    else:
        logger.warning(
            "Using dummy food alternative function. "
            "Please ensure the core module is properly configured with an alternative provider."
        )
        food_alternative = core.dummy_food_alternative

    alternative_output = food_alternative(food_item, k=num_alternatives)

    if alternative_output is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot provide alternatives of '{food_item}' that meet healthiness and sustainability criteria",
        )

    return AlternativeResponse(**alternative_output)
