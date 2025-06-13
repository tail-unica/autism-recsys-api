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


@router.post("/alternative", response_model=AlternativeResponse)
def get_alternative(food_item: str, num_alternatives: int = 5):
    """Alternative food (recipe or ingredient) endpoint. Information about food item is retrieved
    to find alternatives that meet healthiness and sustainability criteria.

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
