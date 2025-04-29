# src/api/main.py (Updated for Prometheus Monitoring)
from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any

# --- Prometheus Client Setup ---
from prometheus_client import Histogram # Import Histogram
# Define a Histogram metric for the closest distance
# Buckets cover distances from 0 to 1 (common for normalized embeddings) + infinity
SEARCH_CLOSEST_DISTANCE = Histogram(
    "search_closest_distance",
    "Distance of the closest search result",
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
)
# --- --- --- --- --- --- --- ---

# --- Configuration ---
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/index/dummy_index.faiss")
MAP_PATH = os.environ.get("FAISS_MAP_PATH", "/app/index/dummy_map.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/metadata/dummy_metadata.pkl")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup Templates ---
templates = Jinja2Templates(directory="src/api/templates")

# --- Load Models and Data (at startup) ---
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

        # --- Observe the closest distance metric ---
        if results: # Only observe if we found at least one result
            closest_distance = results[0]['distance']
            SEARCH_CLOSEST_DISTANCE.observe(closest_distance)
            logger.info(f"Observed closest distance: {closest_distance:.4f}")
        # --- --- --- --- --- --- --- --- --- --- ---

        logger.info(f"Search returned {len(results)} results for query '{query[:100]}...'")
        return results
    except Exception as e:
        logger.error(f"Error during search logic execution: {e}", exc_info=True)
        raise e


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search_browser", response_class=HTMLResponse)
async def search_documents_get_html(request: Request,
                               query: str = Query(..., description="The search query text"),
                               top_k: int = Query(default=3, description="Number of results to return")):
    if not query:
         return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": "Query cannot be empty."})
    try:
        search_results = perform_search(query=query, top_k=top_k)
        return templates.TemplateResponse("results.html", {"request": request, "results": search_results})
    except Exception as e:
        logger.error(f"Error in GET /search_browser endpoint: {e}", exc_info=True)
        return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": "Internal server error during search."})

@app.post("/search")
async def search_documents_post(query: str = Body(...), top_k: int = Body(default=5)):
    try:
        results = perform_search(query=query, top_k=top_k)
        # NOTE: We observe the distance metric only when search is called via GET /search_browser
        # If you want to observe for POST requests too, add the observe logic here as well.
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in POST /search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_name": MODEL_NAME, "faiss_items": index.ntotal}

# --- Prometheus Instrumentation ---
# Import and initialize the instrumentator
from prometheus_fastapi_instrumentator import Instrumentator
try:
    # This attaches default metrics (request count, latency) and exposes /metrics
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus FastAPI instrumentator attached.")
except Exception as e_instr:
     logger.error(f"Failed to attach Prometheus instrumentator: {e_instr}")
# --- --- --- --- --- --- --- ---