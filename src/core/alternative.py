def filter_healthy_and_sustainable(
    alternatives: list[tuple], healthiness: str, sustainability: str, distance_weight: float = 0.8
) -> list:
    """Filter alternatives based on healthiness and sustainability scores.

    Args:
        alternatives (list): List of alternative food items with distances from matched food item.
        healthiness (str): Categorical healthiness score of the matched food item.
        sustainability (str): Categorical sustainability score of the matched food item.

    Returns:
        list: Filtered list of alternatives that are healthier and more sustainable.
    """
    healthiness_reference = map_categorical_score(healthiness)
    sustainability_reference = map_categorical_score(sustainability)

    filtered_alternatives = []
    for alt, dist in alternatives:
        healthiness_score = map_categorical_score(alt.get("healthiness", {}).get("score", None))
        sustainability_score = map_categorical_score(alt.get("sustainability", {}).get("score", None))
        similarity = 1 - dist
        alt_score = alternative_score(
            similarity,
            healthiness_score,
            sustainability_score,
            healthiness_reference,
            sustainability_reference,
            similarity_weight=distance_weight,
        )

        if alt_score is not None:
            filtered_alternatives.append((alt, alt_score))

    if not filtered_alternatives:
        return None

    # Sort alternatives by score in ascending order (lower is better)
    sorted_alternatives = sorted(filtered_alternatives, key=lambda x: x[1])
    filtered_alternatives, filtered_alt_scores = zip(*sorted_alternatives)

    return {"alternatives": filtered_alternatives, "scores": filtered_alt_scores}


def map_categorical_score(score: str) -> float:
    """Map categorical healthiness or sustainability score to a numerical value.

    Args:
        score (str): Categorical score (e.g., 'A', 'B', 'C', 'D', 'E').

    Returns:
        float: Mapped numerical value for the score.
    """
    score_mapping = {
        "A": 1.0,
        "B": 2.0,
        "C": 3.0,
        "D": 4.0,
        "E": 5.0,
        "HIGH": 1.0,
        "MEDIUM": 2.0,
        "LOW": 3.0,
        None: 10.0,
    }
    return score_mapping.get(score, score_mapping[None])


def alternative_score(  # noqa: PLR0913
    similarity: float,
    healthiness: float,
    sustainability: float,
    healthiness_reference: float,
    sustainability_reference: float,
    similarity_weight: float = 0.8,
) -> float:
    """Calculate a score for the alternative based on similarity, healthiness, and sustainability.

    Args:
        similarity (float): Similarity score of the alternative.
        healthiness (float): Healthiness score of the alternative.
        sustainability (float): Sustainability score of the alternative.
        healthiness_reference (float): Reference healthiness score for comparison.
        sustainability_reference (float): Reference sustainability score for comparison.

    Returns:
        float: Combined score for the alternative, or None if the scores are not sufficient.

    """
    healthiness = healthiness - healthiness_reference
    sustainability = sustainability - sustainability_reference
    # Negative scores indicate better healthiness and sustainability trade-off
    if healthiness + sustainability > 0:
        return None
    healthiness, sustainability = abs(healthiness), abs(sustainability)

    return (similarity * similarity_weight + healthiness + sustainability) / 3.0
