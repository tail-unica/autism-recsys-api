FROM python:3.10-slim

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

# Expose port 80 (or choose another port if you prefer)
EXPOSE 8100

# Run the FastAPI application with Uvicorn
CMD ["uv", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8100"]
