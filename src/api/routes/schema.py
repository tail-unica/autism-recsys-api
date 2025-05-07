from typing import Optional

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(description="Unique identifier for the user", example=12345)
    preferences: list[str] = Field(
        description="List of food items, ingredients, or cuisines the user likes",
        example=["pasta", "Italian cuisine", "tomatoes"],
    )
    soft_restrictions: list[str] = Field(
        description="List of food items, ingredients, or cuisines the user dislikes", example=["tomato", "milk"]
    )
    hard_restrictions: Optional[list[str]] = Field(
        default=[],
        description="List of specific food items to completely exclude from recommendations",
        example=["seafood", "peanuts"],
    )
    # conversation_context: Optional[dict[str, Any]] = Field(
    #     default={},
    #     description="Contextual information extracted from the conversation (occasion, mood, etc.)",
    #     example={"occasion": "dinner party", "mood": "celebratory"}
    # )
    meal_time: Optional[str] = Field(
        default=None,
        description="What meal the user is looking for (breakfast, lunch, dinner, snack)",
        example="dinner",
    )
    previous_recommendations: Optional[list[str]] = Field(
        default=[],
        description="List of previously recommended items to avoid repetition",
        example=["spaghetti carbonara", "chicken parmesan"],
    )
    recommendation_count: Optional[int] = Field(
        default=5, description="Number of recommendations to return", example=3
    )
    diversity_factor: Optional[float] = Field(
        default=0.5, description="Controls how diverse the recommendations should be (0.0-1.0)", example=0.7
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier for the conversation these recommendations are associated with",
        example="conv_2025032012345",
    )
