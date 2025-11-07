from typing import Dict, Any, Literal
import numpy.random as random

class DummyGenerator:
    def __init__(self) -> None:
        self.places = [
            "Mole Antonelliana",
            "Birghiotto",
            "Parco del Valentino",
            "Museo Egizio",
            "Piazza Castello",
            "Eataly Torino",
            "Lingotto",
            "Parco della Tesoriera",
        ]
        self.categories = [
            "Museum",
            "Restaurant",
            "Park",
            "Historical Site",
            "Shopping Mall"
        ]
        self.addresses = [
            "Via Montebello, 20, 10124 Torino TO, Italy",
            "Corso Vittorio Emanuele II, 10121 Torino TO, Italy",
            "Piazza Vittorio Veneto, 10123 Torino TO, Italy",
            "Via Accademia delle Scienze, 6, 10123 Torino TO, Italy",
            "Piazza Castello, 10122 Torino TO, Italy",
        ]
        self.sensory_features = [
            "light",
            "space",
            "crowd",
            "noise",
            "odor"
        ]
        self.idiosyncratic_aversions = [
            "bright_light",
            "dim_light",
            "wide_space",
            "narrow_space",
            "crowd",
            "noise",
            "odor",
        ]

    def __call__(self) -> Dict[str, Any]:
        place = random.choice(self.places)
        return {
            "place": place,
            "category": random.choice(self.categories),
            "address": random.choice(self.addresses),
            "sensory_features": {
                "features": [
                    {"feature_name": feature, "rating": random.randint(1, 5)}
                    for feature in self.sensory_features
                ]
            },
        }

def dummy_place_info_fetcher(place: str) -> dict:
    """Dummy place info fetcher."""
    place_info = DummyGenerator()()
    place_info["place"] = place
    return place_info

def dummy_place_recommender(
    user_id: str,
    preferences: tuple[str, ...] = (),
    recommendation_count: int = 5,
    diversity_factor: float = 0.5,
    restrict_preferences: bool = False,
    aversions: tuple[str, ...] = (),
    conversation_id: str | None = None,
) -> dict:
    """Dummy place recommender."""
    generator = DummyGenerator()
    recommendations = []
    for _ in range(recommendation_count):
        place_info = generator()
        recommendations.append(
            {
                "place": place_info["place"],
                "score": random.uniform(0, 1),
                "explanation": f"Recommended because it matches your preferences: {preferences}",
                "metadata": place_info,
            }
        )
    return {
        "user_id": user_id,
        "recommendations": recommendations, # List of tuples (place, score, explanation, metadata)
        "conversation_id": conversation_id,
    }