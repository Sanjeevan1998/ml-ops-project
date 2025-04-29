# src/api/main.py (Updated for Browser GET requests)
from fastapi import FastAPI, HTTPException, Body, Query # Import Query for GET params
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any # For type hinting

# --- Configuration ---
# Uses ENV variables set in Dockerfile (dummy paths)
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/index/dummy_index.faiss")
MAP_PATH = os.environ.get("FAISS_MAP_PATH", "/app/index/dummy_map.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/metadata/dummy_metadata.pkl")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Models and Data (at startup) ---
# (Keep the loading logic the same as before - wrapped in try/except)
try:
    logger.info(f"Loading embedding model: {MODEL_NAME} on device: {MODEL_DEVICE}")
    embedder = SentenceTransformer(MODEL_NAME, device=MODEL_DEVICE)
    logger.info("Embedding model loaded.")

    logger.info(f"Loading FAISS index from: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    logger.info(f"FAISS index loaded. N-items: {index.ntotal}, Dims: {index.d}")

    logger.info(f"Loading FAISS map from: {MAP_PATH}")
    with open(MAP_PATH, "rb") as f:
        faiss_id_to_info = pickle.load(f)
    logger.info(f"FAISS map loaded. Length: {len(faiss_id_to_info)}")

    logger.info(f"Loading metadata store from: {METADATA_PATH}")
    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    logger.info("Metadata store loaded.")

except Exception as e:
    logger.error(f"FATAL: Error loading models/data: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize application: {e}")

app = FastAPI()

# --- Helper Function for Core Search Logic ---
def perform_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Encapsulates the core search logic."""
    try:
        logger.info(f"Performing search: top_k={top_k}, query='{query[:100]}...'")

        query_vector = embedder.encode([query], convert_to_numpy=True, device=MODEL_DEVICE)
        distances, indices = index.search(query_vector.astype(np.float32), min(top_k, index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(faiss_id_to_info): continue

            chunk_info = faiss_id_to_info[idx]
            source_filename = chunk_info.get('source_filename')
            if not source_filename: continue

            doc_metadata = metadata_store.get(source_filename, {})
            distance = distances[0][i]

            results.append({
                "case_name": doc_metadata.get("case_name", "N/A"),
                "citation": doc_metadata.get("citation", "N/A"),
                "source_pdf_filename": source_filename,
                "distance": float(distance),
            })
        logger.info(f"Search returned {len(results)} results for query '{query[:100]}...'")
        return results

    except Exception as e:
        logger.error(f"Error during search logic execution: {e}", exc_info=True)
        # Re-raise or handle appropriately; here we let the endpoint handle HTTP exception
        raise e


# --- API Endpoints ---

# Original POST endpoint (for curl or programmatic access)
@app.post("/search")
async def search_documents_post(query: str = Body(...), top_k: int = Body(default=5)):
    try:
        results = perform_search(query=query, top_k=top_k)
        return {"results": results}
    except Exception as e:
        # Catch exceptions from perform_search or other issues
        logger.error(f"Error in POST /search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")


# NEW GET endpoint (for browser access)
@app.get("/search_browser")
async def search_documents_get(query: str = Query(..., description="The search query text"),
                               top_k: int = Query(default=3, description="Number of results to return")):
    """
    Search endpoint accessible via browser using query parameters.
    Example: /search_browser?query=contract+law&top_k=5
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")
    try:
        results = perform_search(query=query, top_k=top_k)
        return {"results": results}
    except Exception as e:
        # Catch exceptions from perform_search or other issues
        logger.error(f"Error in GET /search_browser endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")


# Health check endpoint (remains the same)
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_name": MODEL_NAME, "faiss_items": index.ntotal}

# --- Prometheus Instrumentation (Keep if needed later, requires library) ---
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)
# logger.info("Prometheus instrumentator attached.")
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---