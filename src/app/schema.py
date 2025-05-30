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


class RecommendationItem(BaseModel):
    """Individual food recommendation with scores and metadata"""

    name: str = Field(description="Name of the recommended food item", example="Spaghetti Carbonara")
    score: float = Field(description="Overall recommendation score", example=0.92)
    explanation: str = Field(
        description="Human-readable explanation of why this item was recommended",
        example=(
            "U25 interacted_with 'Pasta amatriciana' has_ingredient 'guanciale' has_ingredient 'Spaghetti Carbonara'"
        ),
    )
    ingredients: list[tuple[str, str]] = Field(
        description="Main ingredients in this food item",
        example=[("pasta", "100g"), ("eggs", "2"), ("cheese pecorino", "50g"), ("black pepper", "to taste")],
    )
    healthiness_score: float = Field(description="Score indicating healthiness (0.0-1.0)", example=0.65)
    sustainability_score: float = Field(description="Score indicating sustainability (0.0-1.0)", example=0.78)
    nutritional_values: dict[str, float] = Field(
        description="Nutritional information for this food item",
        example={"calories": 450.0, "protein": 12.0, "carbs": 56.0, "fat": 18.0},
    )


class RecommendationResponse(BaseModel):
    """Response model for food recommendations"""

    user_id: int = Field(description="Unique identifier for the user", example=12345)
    recommendations: list[RecommendationItem] = Field(
        description="List of recommended food items with scores and metadata"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier for the conversation these recommendations are associated with",
        example="conv_2025032012345",
    )


class InfoRequest(BaseModel):
    """Request model for food information endpoint"""

    food_item: str = Field(
        description="Name of the food item (recipe or ingredient) to get information about",
        example="Spaghetti Carbonara",
    )


class InfoResponse(BaseModel):
    """Response model for food information endpoint"""

    food_item: str = Field(description="Name of the food item", example="Spaghetti Carbonara")
    healthiness_score: float = Field(description="Score indicating healthiness (0.0-1.0)", example=0.65)
    sustainability_score: float = Field(description="Score indicating sustainability (0.0-1.0)", example=0.78)
    nutritional_values: dict[str, float] = Field(
        description="Nutritional information for this food item",
        example={"calories": 450.0, "protein": 12.0, "carbs": 56.0, "fat": 18.0},
    )
    ingredients: Optional[list[tuple[str, str]]] = Field(
        default=None,
        description="List of main ingredients in this food item if it is a recipe",
        example=[("pasta", "100g"), ("eggs", "2"), ("cheese pecorino", "50g"), ("black pepper", "to taste")],
    )
