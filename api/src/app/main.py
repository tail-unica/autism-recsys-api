from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse

from src import __version__
from src.app.routers import info, recommendation

from src.core.service import RecommenderService

@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RecommenderService()
    await service.initialize()
    app.state.service = service
    try:
        yield
    finally:
        await service.shutdown()

# Create a FastAPI instance
app = FastAPI(
    lifespan=lifespan,
    title = "AutismAPI",
    description = "Place recommendation for People w/ ASD",
    version = __version__
)

app.include_router(info.router, tags=["info"])
app.include_router(recommendation.router, tags=["recommendation"])

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck."""
    if not app.state.service.is_ready():
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy"}