from fastapi import APIRouter, HTTPException

import src.core.main as core
from src.app.schema import InfoResponse

logger = core.logger
router = APIRouter()


@router.get("/food-info/{food_item}", response_model=InfoResponse)
async def get_info(food_item: str):
    """Food information endpoint.

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

    info_response = food_info_fetcher(food_item)

    if info_response is None:
        raise HTTPException(status_code=404, detail=f"Food item '{food_item}' not found in database")

    return InfoResponse(**info_response)
