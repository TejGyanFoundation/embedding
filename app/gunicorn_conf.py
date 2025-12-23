import multiprocessing
import os

# Gunicorn configuration settings
bind = "0.0.0.0:8888"

# Worker configuration
# For CPU-bound tasks (like model inference), it's often better to have fewer workers 
# to avoid context switching overhead and memory usage, especially with large models.
# Adjust 'workers' based on available resources and load testing.
# A safe starting point for ML models is often 1 or 2 workers if GPU is limited or shared.
workers = 1 
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts
# Inference can take time, especially for long texts or large batches
timeout = 120 
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
