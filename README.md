# Indic Sentence Similarity Embedding Service

A high-performance FastAPI service that serves embeddings using the `l3cube-pune/indic-sentence-similarity-sbert` model. This service is containerized with Docker, uses Gunicorn for process management, and manages dependencies with Poetry.

## Features

- **Model**: [l3cube-pune/indic-sentence-similarity-sbert](https://huggingface.co/l3cube-pune/indic-sentence-similarity-sbert)
- **Framework**: FastAPI
- **Process Manager**: Gunicorn with Uvicorn workers
- **Security**: Runs as a non-root user in Docker
- **Packaging**: Docker (Multi-stage build)

## Local Development Setup

### Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/) (Dependency Manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TejGyanFoundation/embedding
    cd embedding
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```
    *Note: If you encounter issues with `package-mode`, ensure `package-mode = false` is set in `pyproject.toml`.*

3.  **Activate the virtual environment:**
    ```bash
    poetry shell
    # OR
    source $(poetry env info --path)/bin/activate
    ```

### Running Locally

To run the server in development mode (with auto-reload):

```bash
poetry run uvicorn app.main:app --port 8888 --reload
```

The API will be available at `http://localhost:8888`.

## Working with Docker

This project includes a production-ready `Dockerfile` and `docker-compose.yml`.

### Build and Run

To build the image and start the container:

```bash
docker-compose up --build
```

The service will be accessible at `http://localhost:8888`.

### Docker Commands

-   **Check Logs:**
    ```bash
    docker-compose logs -f
    ```

-   **Stop Service:**
    ```bash
    docker-compose down
    ```

-   **Check Health Status:**
    The container includes a health check. You can view its status with:
    ```bash
    docker ps
    ```
    Look for `(healthy)` in the STATUS column.

## API Usage

### 1. Health Check

**Endpoint:** `GET /health`

**Request:**
```bash
curl http://localhost:8888/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "l3cube-pune/indic-sentence-similarity-sbert"
}
```

### 2. Generate Embeddings

**Endpoint:** `POST /embed`

**Headers:**
- `Content-Type: application/json`

**Body:**
```json
{
  "text": ["Your text here", "Another sentence"]
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8888/embed" \
     -H "Content-Type: application/json" \
     -d '{"text": ["This is a test sentence", "यह एक परीक्षण वाक्य है"]}'
```

**Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, ...], # Vector for first sentence
    [0.789, 0.012, ...]   # Vector for second sentence
  ]
}
```

## Deployment

The service is designed to be deployed as a Docker container.

### Deploying with Docker

1.  **Build the image:**
    ```bash
    docker build -t embedding-service .
    ```

2.  **Run the container:**
    ```bash
    docker run -d \
      -p 8888:8888 \
      --name embedding-service \
      --restart unless-stopped \
      embedding-service
    ```

### Deploying via Kubernetes (Example)

You can use the built Docker image in a Kubernetes Deployment. Ensure you configure:
-   **Port**: 8888
-   **Liveness/Readiness Probes**: Use the `/health` endpoint.
-   **Resources**: Allocate sufficient CPU/RAM depending on traffic (the model is loaded into memory).

## CI/CD Pipeline

## CI/CD Pipeline

This repository uses a sophisticated GitHub Actions workflow (`.github/workflows/deploy.yml`) that deploys to AWS ECS on pushes to `main`.

### Key Features
- **OIDC Authentication**: Uses AWS IAM OIDC Identity Provider for secure, keyless authentication.
- **Versioning**: Uses date-based tagging (`YYYY.MM.DD.github.RUN_NUMBER.prod`).

### Configuration

To enable the deployment pipeline, you need to configure the following in your GitHub Repository Secrets:

#### AWS
- `AWS_ROLE_ARN`: The ARN of the IAM Role that GitHub Actions will assume.
- `ECR_REPOSITORY_URI`: The full URI of your ECR repository (e.g., `123456789012.dkr.ecr.ap-south-1.amazonaws.com`).
- `AWS_REGION`: (Optional if different from default `ap-south-1`).






