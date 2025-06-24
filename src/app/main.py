from fastapi import FastAPI, Response
from fastapi.responses import RedirectResponse

from src import __version__
from src.app.routers import info, recommendation

# Create a FastAPI instance
app = FastAPI(
    title="PhaseAPI - Food Recommendation System",
    description="API for providing personalized food recommendations",
    version=__version__,
)

app.include_router(info.router, tags=["info"])
app.include_router(recommendation.router, tags=["recommendation"])


@app.get("/", response_class=Response)
async def root():
    """
    Root endpoint that displays links to API documentation and endpoints.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PhaseAPI - Food Recommendation System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            .endpoint {
                background-color: #f8f9fa;
                border-left: 4px solid #4CAF50;
                padding: 10px 15px;
                margin: 15px 0;
                border-radius: 3px;
            }
            a {
                color: #3498db;
                text-decoration: none;
                font-weight: bold;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h1>PhaseAPI - Food Recommendation System</h1>
        <p>Welcome to the Food Recommendation API.
        This service provides personalized food recommendations based on user preferences and restrictions.</p>

        <h2>API Documentation</h2>
        <div class="endpoint">
            <a href="/docs">Swagger UI Documentation</a> - Interactive API documentation
        </div>
        <div class="endpoint">
            <a href="/redoc">ReDoc Documentation</a> - Alternative API documentation
        </div>
        <div class="endpoint">
            <a href="/openapi.json">OpenAPI Schema</a> - Raw OpenAPI specification
        </div>

        <h2>Available Endpoints</h2>
        <div class="endpoint">
            <a href="/recommend">POST /recommend</a> - Get personalized food recommendations
        </div>
        <div class="endpoint">
            <a href="/food-info">GET /food-info</a> - Get information about a specific food item or recipe
        </div>
        <div class="endpoint">
            <a href="/alternative">POST /alternative</a> - Get healtier and more sustainable food alternatives
        </div>

        <p>See the API documentation for details on request and response formats.</p>
    </body>
    </html>
    """
    return html_content


@app.get("/recommend", response_class=RedirectResponse, status_code=307)
async def redirect_to_docs():
    """
    Redirects GET requests to /recommend to the API documentation.
    """
    return "/docs#/recommendation/get_recommendation_recommend_post"


@app.get("/food-info", response_class=RedirectResponse, status_code=307)
async def redirect_to_food_info():
    """
    Redirects GET requests to /food-info to the API documentation.
    """
    return "/docs#/info/get_food_info_food_item_get"


@app.get("/alternative", response_class=RedirectResponse, status_code=307)
async def redirect_to_alternative():
    """
    Redirects GET requests to /alternative to the API documentation.
    """
    return "/docs#/recommendation/get_alternative_recommendation_recommendation_alternative_post"
