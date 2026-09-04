FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
COPY src ./src
COPY demo ./demo

RUN pip install --no-cache-dir . "streamlit==1.62.0"

EXPOSE 8501

CMD ["streamlit", "run", "demo/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
