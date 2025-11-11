FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app

WORKDIR /app

COPY pyproject.toml .
RUN uv sync

EXPOSE 8100

# Run fastapi with uvicorn
CMD ["uv", "run", "python", "-O", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8100"]