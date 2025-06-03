# PHaSE-API - Food Recommendation System

PHaSE-API is a RESTful service that provides personalized food recommendations based on user preferences, restrictions, and other contextual information.

## Getting Started

These instructions will help you set up and run the PHaSE-API on your local machine for development and testing purposes.

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your system

## Running with Docker

### Build the Docker Image

```bash
# Clone the repository
git clone https://github.com/yourusername/PhaseAPI.git
cd PhaseAPI

# Build the Docker image
docker build -t phase-api:latest .
```
## Running with Docker

```bash
# Run the container and map port 8100
docker run -p 8100:8100 phase-api:latest
```

The API will now be accessible at [http://localhost:8100](http://localhost:8100)

## API Documentation

Once the service is running, you can access the API documentation at:

- Swagger UI: [http://localhost:8100/docs](http://localhost:8100/docs)
- ReDoc: [http://localhost:8100/redoc](http://localhost:8100/redoc)
- OpenAPI Schema: [http://localhost:8100/openapi.json](http://localhost:8100/openapi.json)

## Testing Endpoints via Command Line
You can test the API endpoints directly from the command line using tools like _curl_:

### Get API Landing Page
```bash
curl http://localhost:8100/
```

### Get Food Recommendation
```bash
curl -X POST http://localhost:8100/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "preferences": ["pasta", "Italian cuisine", "tomatoes"],
    "soft_restrictions": ["seafood", "spicy"],
    "hard_restrictions": ["peanuts", "shellfish"],
    "meal_time": "dinner",
    "previous_recommendations": ["spaghetti carbonara"],
    "recommendation_count": 3,
    "diversity_factor": 0.7,
    "conversation_id": "conv_2025032012345"
  }'
```

### Get Info about Food Item
```bash
curl -X GET http://localhost:8100/food-info/onion \
  -H "Content-Type: application/json"
```

### Get Alternative Food Items
```bash
curl -X POST http://localhost:8100/alternative \
  -H "Content-Type: application/json" \
  -d '{
    "food_item": "salmon"
  }'
```

## Local Development (without Docker)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Install dependencies using uv, automatically creating a virtual environment
uv sync
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8100
```