from typing import Literal, Optional

from fastapi import APIRouter, HTTPException

import src.core.main as core
from src.app.schema import InfoResponse

logger = core.logger
router = APIRouter()


@router.get("/food-info/{food_item}", response_model=InfoResponse)
async def get_info(
    food_item: str, food_item_type: Optional[Literal["ingredient", "recipe"]] = "ingredient"
) -> InfoResponse:
    """Food information endpoint.

    Multiple food items can have the same name, so this endpoint returns the first match.
    The same name could refer to either a recipe or an ingredient, and without additional context about
    the queried food item, the endpoint cannot distinguish between them.

    **food_item**: Name of the food item (recipe or ingredient) to get information about
    """
    if hasattr(core, "food_info_fetcher"):
        food_info_fetcher = core.food_info_fetcher
    else:
        logger.warning(
            "Using dummy food info fetcher. "
            "Please ensure the core module is properly configured with a food info fetcher."
        )
        food_info_fetcher = core.dummy_food_info_fetcher

    info_response = food_info_fetcher(food_item, food_item_type=food_item_type)

    if info_response is None:
        raise HTTPException(status_code=404, detail=f"Food item '{food_item}' not found in database")

    return InfoResponse(**info_response)
