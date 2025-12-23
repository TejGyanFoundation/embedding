# Builder stage
FROM python:3.13-slim AS builder

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install poetry
RUN pip install --no-cache-dir poetry

COPY pyproject.toml .

# Configure poetry to create venv in project and install dependencies
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-root --no-interaction --no-ansi

# Final stage
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create a non-root user
RUN useradd -m appuser

# Set cache directory for models and change ownership
ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
RUN mkdir -p /app/model_cache && chown -R appuser:appuser /app/model_cache

COPY app ./app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8888

# Add Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8888/health || exit 1

# Run with Gunicorn
CMD ["gunicorn", "-c", "app/gunicorn_conf.py", "app.main:app"]
