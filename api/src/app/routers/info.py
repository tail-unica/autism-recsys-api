from typing import Literal, Optional, List, Dict, Any
import json

from fastapi import APIRouter, Request, Depends, HTTPException, status, Query

from src.app.schema import InfoResponse, SearchRequest, SearchResponse

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

    service._logger.info(f"API get_place_info: response({info_response})")

    if not info_response:
        raise HTTPException(
            status_code=404,
            detail=f"No information found for place '{place}' with the specified model",
        )

    return InfoResponse(**info_response)


@router.post("/search", response_model=SearchResponse)
async def search_places(
    request: SearchRequest,
    service = Depends(_get_service)
) -> SearchResponse:
    """
    Place search endpoint.

    **query**: Search query string
    **limit**: Maximum number of results to return
    **position**: Optional user position for proximity filtering
    **distance**: Optional maximum distance (in meters) from the user position
    **categories**: Optional list of category IDs to filter results
    """
    service._logger.info(f"API search_places: SearchRequest({request})")

    search_results = await service.search_places(query=request.dict())

    service._logger.info(
        "API search_places: response:\n%s",
        json.dumps(
            search_results if not isinstance(search_results, str) else json.loads(search_results),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ),
    )

    return SearchResponse(**search_results)