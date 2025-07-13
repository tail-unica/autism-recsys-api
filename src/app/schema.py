from typing import Literal, Optional

from pydantic import BaseModel, Field


class IngredientList(BaseModel):
    """Model for a list of ingredients"""

    ingredients: list[str] = Field(
        description="List of ingredients",
        example=["pasta", "eggs", "cheese pecorino", "black pepper"],
    )
    quantities: Optional[list[str]] = Field(
        default=None,
        description="Optional list of quantities for each ingredient",
        example=["100g", "2", "50g", "to taste"],
    )


class HealthinessInfo(BaseModel):
    """Model for healthiness information"""

    score: str = Field(
        description="Categorical score indicating healthiness (A - E or Low - HIGH)."
        "It corresponds to the Nutri Score by default",
        example="B",
    )
    qualitative: Optional[str] = Field(
        default=None,
        description="Qualitative description of healthiness score",
        example="Moderate healthiness level",
    )


class SustainabilityInfo(BaseModel):
    """Model for sustainability information"""

    score: str = Field(
        description="""Categorical score across 5 levels indicating sustainability (A - E).
        The score is computed by aggregating the Carbon Footprint (CF) and Water Footprint (WF) estimates
        deriving from the SU-EATABLE-LIFE database. The aggregation is based on the following formula:
        sustainability_score = (cf_score * cf_weight + wf_score * wf_weight)
        where cf_weight and wf_weight are the rescaled factors proposed in the Developer Environmental Footprint
        (EF3.0) package, corresponding to the Climate Change (CF) and Water Use (WF) factors.
        The final scores are then mapped to the categorical scores (A - E) through K-means clustering,
        where A is the most sustainable and E is the least sustainable.
        """,
        example="C",
    )
    qualitative: Optional[str] = Field(
        default=None,
        description="Qualitative description of sustainability score",
        example="Moderate sustainability level",
    )
    CF: Optional[float] = Field(
        default=None,
        description="Carbon Footprint (CF) score",
        example=0.5,
    )
    WF: Optional[float] = Field(
        default=None,
        description="Water Footprint (WF) score",
        example=0.3,
    )


class InfoRequest(BaseModel):
    """Request model for food information endpoint"""

    food_item: str = Field(
        description="Name of the food item (recipe or ingredient) to get information about",
        example="Spaghetti Carbonara",
    )
    food_item_type: Optional[Literal["ingredient", "recipe"]] = Field(
        default=None,
        description="""Type of food item (recipe or ingredient).
        If not provided, the system will determine it automatically""",
        example="recipe",
    )


class InfoResponse(BaseModel):
    """Response model for food information endpoint"""

    food_item: str = Field(description="Name of the food item", example="Spaghetti Carbonara")
    food_item_type: str = Field(description="Type of food item (recipe or ingredient)", example="recipe")
    healthiness: Optional[HealthinessInfo] = Field(
        description="""Healthiness information for this food item, including score and qualitative description""",
        example=HealthinessInfo(score="B", qualitative="Moderate healthiness level"),
    )
    sustainability: Optional[SustainabilityInfo] = Field(
        description="""Sustainability information for this food item, including score,
        qualitative description, and CF/WF scores""",
        example=SustainabilityInfo(score="C", qualitative="Moderate sustainability level", CF=0.5, WF=0.3),
    )
    nutritional_values: Optional[dict[str, Optional[float]]] = Field(
        description="Nutritional information for this food item",
        example={"calories [cal]": 450.0, "protein [g]": 12.0, "carbs [g]": 56.0, "fat [g]": 18.0},
    )
    ingredients: Optional[IngredientList] = Field(
        default=None,
        description="List of ingredients and their quantities for this food item",
        example=IngredientList(
            ingredients=["pasta", "eggs", "cheese pecorino", "black pepper"],
            quantities=["100g", "2", "50g", "to taste"],
        ),
    )
    food_item_url: Optional[str] = Field(
        default=None,
        description="Optional URL to a detailed page about this food item (currently reserved only for recipes)",
        example="https://www.food.com/recipe/cranberry-orange-caramel-corn-447642",
    )


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

    food_item: str = Field(description="Name of the food item", example="Spaghetti Carbonara")
    score: float = Field(description="Overall recommendation score", example=0.92)
    explanation: str = Field(
        description="Human-readable explanation of why this item was recommended",
        example=(
            "U25 interacted_with 'Pasta amatriciana' has_ingredient 'guanciale' has_ingredient 'Spaghetti Carbonara'"
        ),
    )
    food_info: InfoResponse = Field(
        description="Detailed information about the food item",
        example=InfoResponse(
            food_item="Spaghetti Carbonara",
            food_item_type="recipe",
            healthiness=HealthinessInfo(
                score="B",
                qualitative="Good healthiness level",
            ),
            sustainability=SustainabilityInfo(
                score="E", qualitative="Inadequate sustainability level", CF=0.5, WF=0.3
            ),
            nutritional_values={"calories [cal]": 450.0, "protein [g]": 12.0, "carbs [g]": 56.0, "fat [g]": 18.0},
            ingredients=IngredientList(
                ingredients=["pasta", "eggs", "cheese pecorino", "black pepper"],
                quantities=["100g", "2", "50g", "to taste"],
            ),
            food_item_url="https://www.food.com/recipe/spaghetti-carbonara-for-one-447544",
        ),
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


class AlternativeRequest(InfoRequest):
    """Request model for alternative food recommendations"""

    num_alternatives: int = Field(
        default=5,
        description="Number of alternative food items to return",
        example=3,
    )


class AlternativeResponse(BaseModel):
    """Response model for alternative food recommendations"""

    matched_food_item: InfoResponse = Field(
        description="Information about the original food item that was matched",
        example=InfoResponse(
            food_item="Spaghetti Carbonara",
            food_item_type="recipe",
            healthiness=HealthinessInfo(
                score="B",
                qualitative="Good healthiness level",
            ),
            sustainability=SustainabilityInfo(
                score="E", qualitative="Inadequate sustainability level", CF=0.5, WF=0.3
            ),
            nutritional_values={"calories [cal]": 450.0, "protein [g]": 12.0, "carbs [g]": 56.0, "fat [g]": 18.0},
            ingredients=IngredientList(
                ingredients=["pasta", "eggs", "cheese pecorino", "black pepper"],
                quantities=["100g", "2", "50g", "to taste"],
            ),
            food_item_url="https://www.food.com/recipe/spaghetti-carbonara-for-one-447544",
        ),
    )
    alternatives: list[InfoResponse] = Field(
        description="List of alternative food items that meet healthiness and sustainability criteria",
        example=[
            InfoResponse(
                food_item="Pasta alla gricia",
                food_item_type="recipe",
                healthiness=HealthinessInfo(
                    score="A",
                    qualitative="Excellent healthiness level",
                ),
                sustainability=SustainabilityInfo(
                    score="E", qualitative="Inadequate sustainability level", CF=0.5, WF=0.3
                ),
                nutritional_values={"calories [cal]": 400.0, "protein [g]": 15.0, "carbs [g]": 50.0, "fat [g]": 10.0},
                ingredients=IngredientList(
                    ingredients=["pasta", "guanciale", "cheese pecorino", "black pepper"],
                    quantities=["100g", "50g", "30g", "to taste"],
                ),
                food_item_url="https://www.food.com/recipe/pasta-alla-gricia-447544",
            ),
            InfoResponse(
                food_item="Fettuccine Alfredo",
                food_item_type="recipe",
                healthiness=HealthinessInfo(
                    score="C",
                    qualitative="Fair healthiness level",
                ),
                sustainability=SustainabilityInfo(
                    score="B",
                    qualitative="Good sustainability level",
                ),
                nutritional_values={"calories [cal]": 500.0, "protein [g]": 10.0, "carbs [g]": 60.0, "fat [g]": 20.0},
                ingredients=IngredientList(
                    ingredients=["fettuccine", "cream", "parmesan cheese", "butter"],
                    quantities=["100g", "50ml", "30g", "20g"],
                ),
                food_item_url="https://www.food.com/recipe/fettuccine-alfredo-447544",
            ),
            InfoResponse(
                food_item="Penne with basil pesto",
                food_item_type="recipe",
                healthiness=HealthinessInfo(
                    score="B",
                    qualitative="Good healthiness level",
                ),
                sustainability=SustainabilityInfo(
                    score="C",
                    qualitative="Fair sustainability level",
                ),
                nutritional_values={"calories [cal]": 480.0, "protein [g]": 8.0, "carbs [g]": 55.0, "fat [g]": 22.0},
                ingredients=IngredientList(
                    ingredients=["penne", "basil pesto", "olive oil", "parmesan cheese"],
                    quantities=["100g", "30g", "10ml", "20g"],
                ),
                food_item_url="https://www.food.com/recipe/penne-with-basil-pesto-447544",
            ),
        ],
    )
    scores: list[float] = Field(
        description=(
            "List of scores for each alternative indicating how well they meet healthiness and sustainability criteria"
        ),
        example=[0.85, 0.75, 0.65],
    )
