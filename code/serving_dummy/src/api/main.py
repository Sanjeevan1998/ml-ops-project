# src/api/main.py 
from fastapi import FastAPI, HTTPException, Body, Query, Request, Path
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any, Optional # Added Optional
import json
import datetime 
from pydantic import BaseModel
import time # For timing search
import urllib.parse # For decoding filenames

# --- Prometheus Client Setup ---
# ... (Keep Prometheus metrics definitions the same) ...
from prometheus_client import Histogram, Counter
SEARCH_CLOSEST_DISTANCE = Histogram(
    "search_closest_distance",              # Metric name
    "Distance of the closest search result", # Documentation (help text)
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")] # Optional: buckets
)
FEEDBACK_COUNTER = Counter(
    "feedback_received_total",              # Metric name
    "Total count of feedback submissions by type", # Documentation (help text)
    ['feedback_type']                       # Optional: label names
)

# --- Configuration ---
# Paths now loaded from environment variables set in Dockerfile/docker-compose
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/index/real_index.faiss")
MAP_PATH = os.environ.get("FAISS_MAP_PATH", "/app/index/real_map.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/metadata/real_metadata.pkl")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
PDF_DATA_DIR = "/app/pdf_data" # Path *inside container* where real PDFs are mounted

# --- Feedback Logging Setup ---
# ... (Keep feedback model and log file setup the same) ...
class FeedbackItem(BaseModel): ... # Same as before
FEEDBACK_LOG_DIR = "/app/feedback_data"
FEEDBACK_LOG_FILE = os.path.join(FEEDBACK_LOG_DIR, "feedback.jsonl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup Templates ---
templates = Jinja2Templates(directory="src/api/templates")

# --- Load Models and Data ---
embedder = None
index = None
faiss_id_to_info = []
metadata_store = {}
try:
    logger.info(f"Loading embedding model: {MODEL_NAME} on device: {MODEL_DEVICE}")
    embedder = SentenceTransformer(MODEL_NAME, device=MODEL_DEVICE)
    
    logger.info(f"Loading FAISS index from: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    logger.info(f"FAISS index loaded. N-items: {index.ntotal}, Dims: {index.d}")

    logger.info(f"Loading FAISS map from: {MAP_PATH}")
    with open(MAP_PATH, "rb") as f: faiss_id_to_info = pickle.load(f)
    logger.info(f"FAISS map loaded. Length: {len(faiss_id_to_info)}")
    
    logger.info(f"Loading metadata store from: {METADATA_PATH}")
    with open(METADATA_PATH, "rb") as f: metadata_store = pickle.load(f)
    logger.info(f"Metadata store loaded. Keys: {len(metadata_store)}")

    os.makedirs(FEEDBACK_LOG_DIR, exist_ok=True)
    logger.info(f"Feedback log directory ensured: {FEEDBACK_LOG_DIR}")

except Exception as e:
    logger.error(f"FATAL: Error loading models/data: {e}", exc_info=True)
    # Allow app to start maybe, but endpoints will fail - or raise RuntimeError
    # raise RuntimeError(f"Failed to initialize application: {e}") 

app = FastAPI()

# --- Helper Function for Core Search Logic ---
def perform_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    if not embedder or not index or not faiss_id_to_info or not metadata_store:
         logger.error("Search attempted before models/data loaded.")
         raise HTTPException(status_code=503, detail="Service not fully initialized.")
         
    try:
        t_start = time.time()
        logger.info(f"Performing search: top_k={top_k}, query='{query[:100]}...'")
        
        # 1. Embed Query
        query_vector = embedder.encode([query], convert_to_numpy=True, device=MODEL_DEVICE)
        
        # 2. Search FAISS for top N *chunks*
        # Retrieve more chunks initially (e.g., 3*top_k) to allow for aggregation by document
        search_k = min(max(top_k * 3, 10), index.ntotal) # Ensure we search for a reasonable number
        distances, indices = index.search(query_vector.astype(np.float32), search_k)
        
        t_search_done = time.time()
        logger.info(f"FAISS search completed in {t_search_done - t_start:.4f}s. Found {len(indices[0])} potential chunks.")

        # 3. Process results - Aggregate by source document
        doc_results = {} # Key: source_filename, Value: {'best_score': float, 'relevant_chunk': dict}
        
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(faiss_id_to_info): continue # Invalid index

            chunk_info = faiss_id_to_info[idx] # {source_filename, chunk_id, chunk_text}
            source_filename = chunk_info.get('source_filename')
            if not source_filename: continue
            
            distance = distances[0][i]
            # Convert L2 distance to a similarity score (0-1, higher is better)
            # Simple approach: score = 1 / (1 + distance). Max distance is theoretically unbounded.
            # Or, if vectors are normalized (common with sentence-transformers): score = 1 - (distance^2 / 2) approx cosine sim
            similarity_score = 1.0 / (1.0 + float(distance)) # Example score

            # Observe distance of the *absolute* closest chunk found
            if i == 0:
                 SEARCH_CLOSEST_DISTANCE.observe(float(distance))
                 logger.info(f"Observed closest chunk distance: {distance:.4f} (Score: {similarity_score:.4f})")

            # Aggregate: Keep track of the best score found FOR EACH source document
            if source_filename not in doc_results or similarity_score > doc_results[source_filename]['best_score']:
                 doc_results[source_filename] = {
                     'best_score': similarity_score,
                     'relevant_chunk_info': chunk_info, # Store the info of the best chunk for this doc
                     'distance': float(distance) # Store distance corresponding to best score
                 }

        # 4. Sort documents by best score and take top_k
        sorted_docs = sorted(doc_results.items(), key=lambda item: item[1]['best_score'], reverse=True)
        
        # 5. Format final results with metadata
        final_results = []
        for filename, data in sorted_docs[:top_k]:
            doc_metadata = metadata_store.get(filename, {}) # Get placeholder metadata
            chunk_info = data['relevant_chunk_info']
            
            final_results.append({
                "case_name": doc_metadata.get("case_name", "N/A"),
                "citation": doc_metadata.get("citation", "N/A"),
                "decision_date": doc_metadata.get("decision_date", "N/A"),
                "judge": doc_metadata.get("judge", "N/A"),
                "outcome_summary": doc_metadata.get("outcome_summary", "N/A"),
                "key_legal_issues": doc_metadata.get("key_legal_issues", []),
                "source_pdf_filename": filename, 
                "relevant_chunk_text": chunk_info.get('chunk_text', 'N/A'), # Add relevant chunk text
                "similarity_score": data['best_score'],
                "distance": data['distance'] 
            })
            
        t_end = time.time()
        logger.info(f"Search processing finished in {t_end - t_search_done:.4f}s. Returning {len(final_results)} documents.")
        return final_results
        
    except Exception as e:
        logger.error(f"Error during search logic execution: {e}", exc_info=True)
        raise # Re-raise for endpoint handler

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # ... (no change) ...
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search_browser", response_class=HTMLResponse)
async def search_documents_get_html(request: Request,
                               query: str = Query(..., description="The search query text"),
                               top_k: int = Query(default=3, description="Number of results to return")):
    # ... (logic similar, call perform_search, pass results AND query to template) ...
    if not query:
         return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": "Query cannot be empty.", "query": query})
    try:
        search_results = perform_search(query=query, top_k=top_k)
        return templates.TemplateResponse("results.html", {"request": request, "results": search_results, "query": query})
    except HTTPException as http_exc: # Catch specific exceptions if needed
         return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": http_exc.detail, "query": query})
    except Exception as e:
        logger.error(f"Error in GET /search_browser endpoint: {e}", exc_info=True)
        return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": "Internal server error during search.", "query": query})

@app.post("/search")
async def search_documents_post(query: str = Body(...), top_k: int = Body(default=5)):
    # ... (logic similar, call perform_search, return JSON) ...
    try:
        results = perform_search(query=query, top_k=top_k)
        return {"results": results}
    except HTTPException as http_exc:
         raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error in POST /search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")

# --- Endpoint for Logging Feedback ---
@app.post("/log_feedback")
async def log_feedback(payload: FeedbackItem):
    # ... (Keep this endpoint exactly the same as before) ...
    try:
        # ... (increment counter, format log entry, append to file) ...
        feedback_type = payload.feedback.lower()
        FEEDBACK_COUNTER.labels(feedback_type=feedback_type).inc()
        log_entry = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "query": payload.query,
            "source_pdf_filename": payload.source_pdf_filename,
            # Note: The 'distance' logged here will be the distance of the *best chunk* # based on how results.html sends it. Might need adjustment if you want
            # an overall document score logged instead.
            "distance": payload.distance, 
            "feedback": feedback_type
        }
        with open(FEEDBACK_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Feedback logged: {log_entry}")
        return {"status": "success", "message": "Feedback logged"}
    except Exception as e:
        logger.error(f"Error logging feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to log feedback")

# --- NEW Endpoint for Downloading PDFs ---
@app.get("/download/{filename:path}")
async def download_pdf(filename: str = Path(..., description="The name of the PDF file to download")):
    """
    Provides the original PDF file for download.
    Includes basic security check against directory traversal.
    """
    try:
        # **Security:** Decode filename and prevent directory traversal
        decoded_filename = urllib.parse.unquote(filename)
        # Basic sanitization: Check for ".." or "/" characters that could escape the intended directory
        if ".." in decoded_filename or decoded_filename.startswith("/") or "\\" in decoded_filename:
             logger.warning(f"Attempted directory traversal: {filename}")
             raise HTTPException(status_code=400, detail="Invalid filename")

        # Construct the full path *within the container*
        file_path = os.path.join(PDF_DATA_DIR, decoded_filename)
        file_path = os.path.abspath(file_path) # Normalize path

        # Extra check: Ensure the final path is still within the PDF_DATA_DIR
        if not file_path.startswith(os.path.abspath(PDF_DATA_DIR)):
             logger.warning(f"Attempted directory traversal after normalization: {filename} -> {file_path}")
             raise HTTPException(status_code=400, detail="Invalid filename")
             
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"Download request for non-existent file: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")

        logger.info(f"Providing download for: {file_path}")
        # Return the file as a response
        return FileResponse(
            path=file_path,
            filename=decoded_filename, # Suggest original filename to browser
            media_type='application/pdf'
        )
    except HTTPException as http_exc:
         raise http_exc # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error during file download for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during download")
# --- --- --- --- --- --- --- --- --- --- ---

@app.get("/health")
async def health_check():
    # ... (no change, maybe update faiss_items source if index object name changes) ...
    status = "ok"
    items = "N/A"
    if embedder and index and faiss_id_to_info and metadata_store:
         items = index.ntotal
    else:
         status = "degraded - models/data not fully loaded"
         
    return {"status": status, "model_name": MODEL_NAME, "faiss_items": items}

# --- Prometheus Instrumentation ---
# ... (Keep this the same) ...
from prometheus_fastapi_instrumentator import Instrumentator
try:
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus FastAPI instrumentator attached.")
except Exception as e_instr:
     logger.error(f"Failed to attach Prometheus instrumentator: {e_instr}")