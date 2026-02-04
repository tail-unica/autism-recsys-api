FROM python:3.10-slim

# NOTE: remove git installation after hopwise is published to PyPI
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --upgrade pip
RUN uv pip install dgl>=2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps
RUN uv pip install --system -p python3.10 .
# -- or instead of pip install
# RUN uv sync

# ENV DGLBACKEND=pytorch