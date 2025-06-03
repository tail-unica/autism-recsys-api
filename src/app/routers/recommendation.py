from fastapi import APIRouter

from src.app.schema import (
    AlternativeResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from src.core.main import food_alternative_recommender, food_info_fetcher, food_recommender

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
def get_alternative_recommendation(food_item: str):
    """Alternative food (recipe or ingredient) recommendation endpoint. Information about food item is retrieved
    to find alternatives that meet healthiness and sustainability criteria.

    **food_item**: Name of the food item (recipe or ingredient) to get alternative recommendations for
    """

    info_response = food_info_fetcher(food_item)

    alternatives = food_alternative_recommender(**info_response)

    alternative_response = AlternativeResponse(food_item=food_item, alternatives=alternatives)

    return alternative_response
