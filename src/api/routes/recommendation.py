from fastapi import APIRouter

from src.api.routes.schema import RecommendationRequest
from src.core.main import food_recommender

router = APIRouter()


@router.post("/recommend")
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
    recommendations, scores = food_recommender(
        request.user_id,
        preferences=request.preferences,
        soft_restrictions=request.soft_restrictions,
        hard_restrictions=request.hard_restrictions,
        previous_recommendations=request.previous_recommendations,
        meal_time=request.meal_time,
        recommendation_count=request.recommendation_count,
        diversity_factor=request.diversity_factor,
    )

    return {
        "user_id": request.user_id,
        "recommendations": recommendations,
        "scores": scores,
        "conversation_id": request.conversation_id,
    }
