from typing import Literal, Optional

from fastapi import APIRouter, Request, Depends, HTTPException, status

from src.app.schema import InfoResponse

router = APIRouter()

def _get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if not service or not service.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service warming up")
    return service

@router.get("/place-info/{place}", response_model=InfoResponse)
async def get_place_info(
    place: str,
    service = Depends(_get_service)
) -> InfoResponse:
    """
    Place information endpoint.

    Multiple places can have the same name, so this endpoint returns the first match.
    The same name could refer to different types of places, and without additional context about
    the queried place, the endpoint cannot distinguish between them.

    **place**: Name of the place to get information about
    **model**: Model to use for fetching information
    """
    service._logger.info(f"API get_place_info: place({place})")
    info_response = await service.fetch_place_info(place)

    if not info_response:
        raise HTTPException(
            status_code=404,
            detail=f"No information found for place '{place}' with the specified model",
        )

    return InfoResponse(**info_response)