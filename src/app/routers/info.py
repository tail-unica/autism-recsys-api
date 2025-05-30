from fastapi import APIRouter

from src.app.schema import InfoRequest, InfoResponse

router = APIRouter()


@router.post("/food-info", response_model=InfoResponse)
async def get_info(request: InfoRequest):
    """Food information endpoint.

    **food_item**: Name of the food item (recipe or ingredient) to get information about
    """
    # Here you would typically call a function to fetch food information
    # For demonstration, we return a mock response
    info_response = {
        "food_item": request.food_item,
        "description": f"Information about {request.food_item}",
        "nutritional_values": {
            "calories": 100,
            "protein": 5,
            "carbohydrates": 20,
            "fats": 2,
        },
        "health_benefits": f"Health benefits of {request.food_item}",
    }

    return InfoResponse(**info_response)
