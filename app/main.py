from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Indic Sentence Similarity API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model globally
MODEL_NAME = "l3cube-pune/indic-sentence-similarity-sbert"
model = None
init_error = None

def load_model():
    global model, init_error
    if model is not None:
        return True, "Model already loaded"
    try:
        logger.info(f"Loading model: {MODEL_NAME}...")
        
        model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded successfully.")
        init_error = None
        return True, "Model loaded successfully"
    except Exception as e:
        logger.error(f"Error loading model {MODEL_NAME}: {e}")
        init_error = str(e)
        return False, str(e)

@app.on_event("startup")
async def startup_event():
    load_model()

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

@app.post("/init-model")
def initialize_model():
    success, message = load_model()
    if success:
        return {"status": "ok", "message": message}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {message}")

@app.get("/health")
def health_check():
    if model is None:
         return {"status": "error", "message": "Model not loaded", "error": init_error}
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/embed", response_model=EmbeddingResponse)
def get_embeddings(request: EmbeddingRequest):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not initialized. Error: {init_error}")
    
    sentences = request.text
    if isinstance(sentences, str):
        sentences = [sentences]
    
    try:
        embeddings = model.encode(sentences)
        return EmbeddingResponse(embeddings=embeddings.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
