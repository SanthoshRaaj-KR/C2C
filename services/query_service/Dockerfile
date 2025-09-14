FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p query_chroma_db

EXPOSE 8003

CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8003"]