from typing import Generic, Optional, List, Set, TypeVar
from pydantic import BaseModel, Field, PositiveInt, model_validator
from pydantic.config import ConfigDict
from enum import Enum

EnumType = TypeVar("EnumType", bound=Enum)


# --- Enums ---

class SensoryFeatures(str, Enum):
    """Enum for sensory features levels"""

    LIGHT = "light"
    SPACE = "space"
    CROWD = "crowd"
    NOISE = "noise"
    ODOR = "odor"


class IdiosyncraticAversions(str, Enum):
    """Enum for idiosyncratic aversions levels"""

    BRIGHT_LIGHT = "bright_light"
    DIM_LIGHT = "dim_light"
    WIDE_SPACE = "wide_space"
    NARROW_SPACE = "narrow_space"
    CROWD = "crowd"
    NOISE = "noise"
    ODOR = "odor"


# --- Generic Sensory Feature Set ---

class Feature(BaseModel, Generic[EnumType]):
    """Sensory feature rating for an item"""

    feature_name: EnumType = Field(
        description="Name of the sensory feature",
        example="light",
    )
    rating: int = Field(
        description="Rating of the sensory feature from 1 (low) to 5 (high)",
        example=3,
        ge=1,
        le=5,
    )


class FeatureSet(BaseModel, Generic[EnumType]):
    """Set of sensory features with ratings"""

    features: List[Feature[EnumType]] = Field(
        description="List of sensory features with their ratings",
        example=[
            {"feature_name": "light", "rating": 3},
            {"feature_name": "space", "rating": 4},
        ],
    )

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="after")
    def validate_features(self) -> "FeatureSet[EnumType]":
        if not self.features:
            raise ValueError("At least one feature must be provided.")

        enum_cls = self.__pydantic_generic_metadata__["args"][0]

        required: Set[EnumType] = set(enum_cls)
        provided: Set[EnumType] = {f.feature_name for f in self.features}

        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing required features: {', '.join(m.value for m in missing)}"
            )

        extras = provided - required
        if extras:
            raise ValueError(
                f"Unexpected features: {', '.join(m.value for m in extras)}"
            )

        return self


# --- Request and Response Models ---

class InfoRequest(BaseModel):
    """Request model for place information endpoint"""

    place: str = Field(
        description="Name of the place to get information about", example="Mole Antonelliana"
    )


class InfoResponse(BaseModel):
    """Response model for place information endpoint"""

    place: str = Field(
        description="Name of the place", example="Mole Antonelliana"
    )
    category: Optional[str] = Field(
        default=None, description="Category of the place", example="Museum"
    )
    address: Optional[str] = Field(
        default=None, description="Address of the place", example="Via Montebello, 20, 10124 Torino TO, Italy"
    )
    sensory_features: FeatureSet[SensoryFeatures] = Field(
        description="Sensory features ratings for the place",
        example={
            "features": [
                {"feature_name": "light", "rating": 3},
                {"feature_name": "space", "rating": 4},
                {"feature_name": "crowd", "rating": 2},
                {"feature_name": "noise", "rating": 3},
                {"feature_name": "odor", "rating": 1},
            ]
        },
    )


class RecommendationRequest(BaseModel):
    """Request model for place recommendations"""

    user_id: str = Field(
        description="Unique identifier for the user", 
        example="12345"
    )
    preferences: List[str] = Field(
        description="List of user preferences for places", 
        example=["Monte dei Cappuccini", "Mercato"]
    )
    recommendation_count: Optional[PositiveInt] = Field(
        default=5,
        description="Number of recommendations to return", 
        example=3
    )
    diversity_factor: Optional[float] = Field(
        default=0.5,
        description="Diversity factor for recommendations (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.7,
    )
    restrict_preferences: Optional[bool] = Field(
        default=False,
        description="Whether to restrict recommendations based on user preferences",
        example=True,
    )
    aversions: FeatureSet[IdiosyncraticAversions] = Field(
        description="Sensory idiosyncratic aversion details for the user",
        example={
            "features": [
                {"feature_name": "bright_light", "rating": 2},
                {"feature_name": "dim_light", "rating": 3},
                {"feature_name": "wide_space", "rating": 1},
                {"feature_name": "narrow_space", "rating": 4},
                {"feature_name": "crowd", "rating": 5},
                {"feature_name": "noise", "rating": 2},
                {"feature_name": "odor", "rating": 1},
            ]
        },
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier for the conversation these recommendations are associated with",
        example="conv_2025032012345",
    )


class RecommendationItem(BaseModel):
    """Individual recommendation item with score and metadata"""

    place: str = Field(description="Name of the recommended place", example="Egyptian Museum")
    score: float = Field(
        description="Recommendation score indicating suitability",
        example=0.95,
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of why this item was recommended",
    )
    metadata: InfoResponse = Field(
        description="Additional information about the recommended place"
    )


class RecommendationResponse(BaseModel):
    """Response model for food recommendations"""

    user_id: str = Field(description="Unique identifier for the user", example="12345")
    recommendations: list[RecommendationItem] = Field(
        description="List of recommended food items with scores and metadata"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier for the conversation these recommendations are associated with",
        example="conv_2025032012345",
    )
