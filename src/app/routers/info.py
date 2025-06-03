from fastapi import APIRouter

from src.app.schema import InfoResponse, IngredientList
from src.core.main import food_info_fetcher

router = APIRouter()


@router.get("/food-info/{food_item}", response_model=InfoResponse)
async def get_info(food_item: str):
    """Food information endpoint.

    **food_item**: Name of the food item (recipe or ingredient) to get information about
    """
    info_response = food_info_fetcher(food_item)

    # [("ingredient", "quantity"), ...] -> (["ingredient1", "ingredient2", ...], ["quantity1", "quantity2", ...])
    if info_response.get("ingredients") is not None:
        ingredients, quantities = zip(*info_response["ingredients"])
        ingredients, quantities = list(ingredients), list(quantities)
    else:
        ingredients, quantities = [], []

    response = InfoResponse(
        food_item=info_response["food_item"],
        healthiness_score=info_response["healthiness_score"],
        sustainability_score=info_response["sustainability_score"],
        nutritional_values=info_response["nutritional_values"],
        ingredients=IngredientList(ingredients=ingredients, quantities=quantities),
    )

    return response
