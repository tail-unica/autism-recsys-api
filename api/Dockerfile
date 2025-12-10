FROM python:3.10-slim

# Install git and ca-certificates
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv from Astral's UV image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project

COPY . .

RUN uv sync --frozen

EXPOSE 8100

# Run fastapi with uvicorn
CMD ["uv", "run", "python", "-O", "-m", "uvicorn", "src.app.main:app", "--log-level", "debug", "--host", "0.0.0.0", "--port", "8100"]