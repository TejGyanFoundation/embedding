from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import os
import logging
import threading

# Serialize model.encode calls: concurrent encodes multiply peak memory
# (each holds activation tensors for its whole batch) and under memory
# pressure the process balloons and thrashes swap until nothing completes.
# One encode at a time keeps the footprint flat; queued requests just wait.
_encode_lock = threading.Lock()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Indic Sentence Similarity API")

from app.pdf_import import router as pdf_import_router  # noqa: E402

app.include_router(pdf_import_router)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model globally
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
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
        with _encode_lock:
            embeddings = model.encode(sentences)
        return EmbeddingResponse(embeddings=embeddings.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)


class SimilarTermsRequest(BaseModel):
    query: str
    texts: List[str]
    top_n: int = 4
    min_score: float = 0.42


class SimilarTermsResponse(BaseModel):
    terms: List[List[str]]


# Devanagari letters/matras only (danda ।॥ and digits excluded), or Latin words.
_WORD_RE = __import__("re").compile(r"[\u0900-\u0963\u0970-\u097F]{3,}|[A-Za-z]{4,}")

# Hindi function words: they embed close to everything and explain nothing.
_STOPWORDS = frozenset(
    """है हैं हो हूँ हूं था थी थे और आप तुम हम वे यह वह जो तो भी ही ना नहीं मत
    के का की को से में पर इस उस ऐसा वैसा कैसा कोई कुछ सब हर एक अपना अपनी अपने
    गया गई गए हुआ हुई हुए रहा रही रहे कर करो करें किया करना वाला वाली वाले
    लिए साथ बाद पहले अभी यहाँ वहाँ कहाँ क्या क्यों जब तब अगर मगर लेकिन तथा
    उसे इसे मुझे तुम्हें हमें उनको इनको मेरा तेरा उसका इसका जिसका""".split()
)
# Query-vs-word similarity cache: Hindi corpora reuse a small vocabulary, so
# repeated searches hit the cache almost entirely.
_word_vector_cache: dict = {}
_WORD_CACHE_MAX = 50000


@app.post("/similar-terms", response_model=SimilarTermsResponse)
def similar_terms(request: SimilarTermsRequest):
    """For each text, return the words most semantically similar to the query.

    Used by search to show WHY a section matched in semantic mode: the
    returned words get highlighted even when the literal query words are
    absent from the text.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    import numpy as np

    per_text_words = []
    unique_words: list = []
    seen = set()
    for text in request.texts:
        words = []
        for match in _WORD_RE.finditer(text or ""):
            word = match.group(0)
            key = word.casefold()
            if key in _STOPWORDS:
                continue
            if key not in seen:
                seen.add(key)
                unique_words.append(word)
            words.append(word)
        per_text_words.append(words)

    to_embed = [w for w in unique_words if w.casefold() not in _word_vector_cache]
    if to_embed:
        with _encode_lock:
            vectors = model.encode(to_embed, show_progress_bar=False)
        for word, vector in zip(to_embed, vectors):
            if len(_word_vector_cache) < _WORD_CACHE_MAX:
                _word_vector_cache[word.casefold()] = np.asarray(
                    vector, dtype=np.float32
                )

    with _encode_lock:
        query_vector = np.asarray(
            model.encode([request.query])[0], dtype=np.float32
        )
    query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)

    def score(word: str) -> float:
        vector = _word_vector_cache.get(word.casefold())
        if vector is None:
            return 0.0
        return float(
            np.dot(query_norm, vector / (np.linalg.norm(vector) + 1e-9))
        )

    results = []
    for words in per_text_words:
        scored = {}
        for word in words:
            key = word.casefold()
            if key in _STOPWORDS:
                continue
            if key not in scored:
                scored[key] = (score(word), word)
        if not scored:
            results.append([])
            continue
        best = max(pair[0] for pair in scored.values())
        # Absolute floor plus a relative gate: only words nearly as close to
        # the query as the text's best word count as "what it matched".
        threshold = max(request.min_score, 0.78 * best)
        top = sorted(
            (pair for pair in scored.values() if pair[0] >= threshold),
            reverse=True,
        )[: request.top_n]
        results.append([word for _, word in top])

    return SimilarTermsResponse(terms=results)
