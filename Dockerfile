FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cli/ ./cli/
RUN pip install --no-cache-dir ./cli

COPY demo-api/requirements.txt ./demo-api/requirements.txt
RUN pip install --no-cache-dir -r demo-api/requirements.txt

COPY demo-api/ ./demo-api/

EXPOSE 8000
WORKDIR /app/demo-api
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

