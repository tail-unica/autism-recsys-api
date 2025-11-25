FROM python:3.10-slim

WORKDIR /src 

ENV CONFIG_DATASET_PATH=/config
ENV CONFIG_DATASET_NAME="dataset"

COPY pyproject.toml .
RUN pip install .

# CMD ["python", "main.py"]

# docker build -t python-exp .
# docker run -it --rm -v $(pwd):/src -w /src python-exp bash
# docker run -it --rm -v $(pwd):/src -w /src python-exp

# ---- COMPOSE ----
# docker compose exec create_db bash -c "python3 main.py"
