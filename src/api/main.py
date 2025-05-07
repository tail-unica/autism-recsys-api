from fastapi import FastAPI

from src.api.routes import recommendation_router

# Create a FastAPI instance
app = FastAPI()

app.include_router(recommendation_router)
