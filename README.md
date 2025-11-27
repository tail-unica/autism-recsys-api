# Autism-recsys-API

Autism-recsys-API is an API service that provides recommendations for points of interest (POIs) tailored for individuals with autism. The API leverages machine learning models to deliver personalized recommendations based on user preferences and behaviors.

## Getting Started

These instructions will help you set up and run the PHaSE-API on your local machine for development and testing purposes.

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your system
- [Pretrained Weights and Dataset]() **TODO**: will use a docker volume to mount these files

## Running with Docker

### Build the Docker Image

```bash
# Clone the repository
git clone https://github.com/tail-unica/autism-recsys-api.git
cd autism-recsys-api

# Build the Docker Compose
docker compose build
```
## Running with Docker

```bash
# Run the container and map port 8100
docker compose up
```

GPU support is enabled by default, enabled by using the following command (if it doesn't work, ensure you have the NVIDIA Container Toolkit installed):

```bash
# TODO
docker compose up --gpus all
docker compose up --gpus '"device=1"'  # to use a specific GPU
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

## Local Development (without Docker)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Install dependencies using uv, automatically creating a virtual environment
uv sync
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the application
uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8100
```