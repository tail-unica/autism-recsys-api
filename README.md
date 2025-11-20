# PHaSE-API - Food Recommendation System

PHaSE-API is a RESTful service that provides personalized food recommendations based on user preferences, restrictions, and other contextual information.

## Getting Started

These instructions will help you set up and run the PHaSE-API on your local machine for development and testing purposes.

### Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed on your system
- [Pretrained Weights and Dataset](https://drive.google.com/drive/folders/1Vwa2Fje6Ltn-sNo6eaZSpRhgrEviSfzW?usp=sharing) downloaded and placed at the root of the project directory. It includes two folders:
  - `checkpoint`: Contains the pretrained model weights, the preprocessed dataset, and the tokenizer.
  - `data`: Contains the dataset used to retrieve food information.

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
docker run -p 8100:8100 phase-api:latest
```

GPU support is enabled by default, enabled by using the following command (if it doesn't work, ensure you have the NVIDIA Container Toolkit installed):

```bash
docker run --gpus all -p 8100:8100 phase-api:latest
docker run --gpus '"device=1"' -p 8100:8100 phase-api:latest  # to use a specific GPU
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