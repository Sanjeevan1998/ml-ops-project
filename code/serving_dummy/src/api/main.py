# src/api/main.py (Cleaned Up)
from fastapi import FastAPI, HTTPException, Body, Query, Request, Path, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse # Removed Response as it wasn't used directly
from fastapi.templating import Jinja2Templates
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from typing import List, Dict, Any, Optional 
import json
import datetime 
from pydantic import BaseModel
import time 
import urllib.parse 
import fitz # PyMuPDF
import re   
import io   

# --- Prometheus Client Setup ---
from prometheus_client import Histogram, Counter
SEARCH_CLOSEST_DISTANCE = Histogram(
    "search_closest_distance",              
    "Distance of the closest search result", 
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")] 
)
FEEDBACK_COUNTER = Counter(
    "feedback_received_total",              
    "Total count of feedback submissions by type", 
    ['feedback_type']                       
)

# --- Configuration ---
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/index/real_index.faiss")
MAP_PATH = os.environ.get("FAISS_MAP_PATH", "/app/index/real_map.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/metadata/real_metadata.pkl")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu")
PDF_DATA_DIR = "/app/pdf_data" 

# --- Feedback Logging Setup ---
class FeedbackItem(BaseModel): # Define structure for feedback payload
    query: str
    source_pdf_filename: str
    distance: float
    feedback: str 
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
# Constants for chunking (Consider making these env vars)
CHUNK_SIZE = 400  
CHUNK_OVERLAP = 50

try:
    # ... (Loading logic remains the same) ...
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
    # Application will likely fail later if loading fails

app = FastAPI()

# --- Helper Functions ---

def simple_chunker(text, chunk_size, chunk_overlap):
    # ... (Chunker logic remains the same) ...
    words = re.split(r'(\s+)', text) 
    tokens = [word for word in words if word.strip()] 
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunks.append(" ".join(chunk_tokens))
        step = chunk_size - chunk_overlap
        if step <= 0: step = max(1, chunk_size // 2) 
        start_idx += step
        if end_idx == len(tokens): break
    return chunks

def search_index_with_vector(query_vector: np.ndarray, search_k: int) -> tuple[np.ndarray, np.ndarray]:
    # ... (Logic remains the same) ...
     if not index: raise HTTPException(status_code=503, detail="FAISS Index not loaded.")
     effective_k = min(search_k, index.ntotal)
     if effective_k <= 0: return (np.array([]), np.array([])) 
     return index.search(query_vector.astype(np.float32), effective_k)

def aggregate_results_by_doc(distances: np.ndarray, indices: np.ndarray) -> Dict[str, Dict[str, Any]]:
    # ... (Logic remains the same) ...
    doc_results = {} 
    if indices.size == 0: return doc_results 
    for i, idx in enumerate(indices[0]):
        if idx == -1 or idx >= len(faiss_id_to_info): continue 
        chunk_info = faiss_id_to_info[idx] 
        source_filename = chunk_info.get('source_filename')
        if not source_filename: continue
        distance = distances[0][i]
        similarity_score = 1.0 / (1.0 + float(distance)) 
        if source_filename not in doc_results or similarity_score > doc_results[source_filename]['best_score']:
             doc_results[source_filename] = {
                 'best_score': similarity_score,
                 'relevant_chunk_info': chunk_info, 
                 'distance': float(distance) 
             }
    return doc_results

def format_final_results(sorted_docs: list, top_k: int) -> List[Dict[str, Any]]:
     # ... (Logic remains the same) ...
     final_results = []
     for filename, data in sorted_docs[:top_k]:
         doc_metadata = metadata_store.get(filename, {}) 
         chunk_info = data['relevant_chunk_info']
         final_results.append({
             "case_name": doc_metadata.get("case_name", "N/A"),
             "citation": doc_metadata.get("citation", "N/A"),
             "decision_date": doc_metadata.get("decision_date", "N/A"),
             "judge": doc_metadata.get("judge", "N/A"),
             "outcome_summary": doc_metadata.get("outcome_summary", "N/A"),
             "key_legal_issues": doc_metadata.get("key_legal_issues", []),
             "source_pdf_filename": filename, 
             "relevant_chunk_text": chunk_info.get('chunk_text', 'N/A'),
             "similarity_score": data['best_score'],
             "distance": data['distance'] 
         })
     return final_results

# --- Combined Search Logic Function ---
async def perform_combined_search(query: Optional[str], query_file: Optional[UploadFile], top_k: int) -> List[Dict[str, Any]]:
    # ... (Combined search logic as implemented previously remains the same) ...
    if not embedder or not index or not faiss_id_to_info or not metadata_store:
         logger.error("Search attempted before models/data loaded.")
         raise HTTPException(status_code=503, detail="Service not fully initialized.")
    if not query and not query_file:
         raise HTTPException(status_code=400, detail="Please provide a text query or upload a file.")

    t_start_total = time.time()
    all_doc_results_detailed = {} # Store detailed results { filename: {'best_score': ..., 'relevant_chunk_info': ..., 'distance': ...} }

    # --- 1. Process Text Query (if provided) ---
    if query:
        t_start_text = time.time()
        logger.info(f"Processing text query: '{query[:100]}...'")
        try:
            query_vector = embedder.encode([query], convert_to_numpy=True, device=MODEL_DEVICE)
            search_k_text = min(max(top_k * 2, 10), index.ntotal) 
            distances, indices = search_index_with_vector(query_vector, search_k_text)
            doc_results_text = aggregate_results_by_doc(distances, indices) 
            
            # Merge into detailed results, keeping the best score found so far
            for filename, data in doc_results_text.items():
                if filename not in all_doc_results_detailed or data['best_score'] > all_doc_results_detailed[filename]['best_score']:
                     all_doc_results_detailed[filename] = data
            
            logger.info(f"Text query processing took {time.time() - t_start_text:.4f}s. Found {len(doc_results_text)} potential docs.")
            if indices.size > 0 and indices[0][0] != -1:
                 SEARCH_CLOSEST_DISTANCE.observe(float(distances[0][0]))
        except Exception as e:
             logger.error(f"Error processing text query: {e}", exc_info=True)

    # --- 2. Process File Query (if provided) ---
    file_chunks = [] # Keep track of file chunks if processed
    if query_file and query_file.filename:
         t_start_file = time.time()
         logger.info(f"Processing uploaded file: {query_file.filename}")
         if not query_file.filename.lower().endswith(".pdf"):
               logger.warning(f"Invalid file type uploaded: {query_file.filename}")
               # Potentially raise HTTPException or just ignore the file
         else:
             try:
                  pdf_content = await query_file.read()
                  await query_file.close() 
                  doc_text = ""
                  with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
                       for page in doc: doc_text += page.get_text() + "\n"
                  
                  if not doc_text.strip():
                       logger.warning(f"Uploaded file {query_file.filename} contains no extractable text.")
                  else:
                       file_chunks = simple_chunker(doc_text, CHUNK_SIZE, CHUNK_OVERLAP) # Use constants
                       logger.info(f"Split uploaded file into {len(file_chunks)} chunks.")

                       if file_chunks:
                           chunk_embeddings = embedder.encode(file_chunks, convert_to_numpy=True, device=MODEL_DEVICE, show_progress_bar=False) 
                           search_k_file = min(max(top_k // len(file_chunks) + 1, 3), index.ntotal) if len(file_chunks) > 0 else 3
                           
                           for vec in chunk_embeddings:
                                distances_f, indices_f = search_index_with_vector(np.array([vec]), search_k_file)
                                doc_results_chunk = aggregate_results_by_doc(distances_f, indices_f)
                                # Merge into detailed results, keeping the best score found so far
                                for filename, data in doc_results_chunk.items():
                                     if filename not in all_doc_results_detailed or data['best_score'] > all_doc_results_detailed[filename]['best_score']:
                                          all_doc_results_detailed[filename] = data
                  logger.info(f"File query processing took {time.time() - t_start_file:.4f}s.")
             except Exception as e:
                  logger.error(f"Error processing uploaded file {query_file.filename}: {e}", exc_info=True)

    # --- 3. Combine and Format Final Results ---
    if not all_doc_results_detailed:
         logger.info("No results found from either text or file query.")
         return [] 

    # Sort the combined detailed results by score
    sorted_detailed_docs = sorted(all_doc_results_detailed.items(), key=lambda item: item[1]['best_score'], reverse=True)

    # Format final list
    final_results = format_final_results(sorted_detailed_docs, top_k)

    logger.info(f"Combined search finished in {time.time() - t_start_total:.4f}s. Returning {len(final_results)} documents.")
    return final_results


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint for combined search (handles form POST)
@app.post("/search_combined", response_class=HTMLResponse)
async def search_combined_post_html(
    request: Request,
    top_k: int = Form(default=5),
    query: Optional[str] = Form(None), 
    query_file: Optional[UploadFile] = File(None) 
):
    # Determine context for display (text query or filename)
    display_query = query if query else (query_file.filename if query_file and query_file.filename else "N/A")
    try:
        search_results = await perform_combined_search(query=query, query_file=query_file, top_k=top_k)
        return templates.TemplateResponse("results.html", {"request": request, "results": search_results, "query": display_query})
    except HTTPException as http_exc: 
         return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": http_exc.detail, "query": display_query})
    except Exception as e:
        logger.error(f"Error in POST /search_combined endpoint: {e}", exc_info=True)
        return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": "Internal server error during combined search.", "query": display_query})

# Endpoint for programmatic access (handles text query only via JSON body)
@app.post("/search") 
async def search_documents_post(query: str = Body(...), top_k: int = Body(default=5)):
    try:
         # Calls the combined search logic, passing None for the file
         search_results_detailed = await perform_combined_search(query=query, query_file=None, top_k=top_k)
         return {"results": search_results_detailed} 
    except HTTPException as http_exc:
         raise http_exc 
    except Exception as e:
        logger.error(f"Error in POST /search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")

# --- Feedback Endpoint ---
@app.post("/log_feedback")
async def log_feedback(payload: FeedbackItem):
    # ... (Logic remains the same) ...
    try:
        feedback_type = payload.feedback.lower()
        FEEDBACK_COUNTER.labels(feedback_type=feedback_type).inc()
        log_entry = { /* ... log entry details ... */ }
        with open(FEEDBACK_LOG_FILE, "a") as f: f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Feedback logged: {log_entry}")
        return {"status": "success", "message": "Feedback logged"}
    except Exception as e:
        logger.error(f"Error logging feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to log feedback")


# --- Download Endpoint ---
@app.get("/download/{filename:path}")
async def download_pdf(filename: str = Path(..., description="The name of the PDF file to download")):
    # ... (Logic remains the same, including security checks) ...
    try:
        decoded_filename = urllib.parse.unquote(filename)
        if ".." in decoded_filename or decoded_filename.startswith("/") or "\\" in decoded_filename:
             raise HTTPException(status_code=400, detail="Invalid filename")
        file_path = os.path.abspath(os.path.join(PDF_DATA_DIR, decoded_filename))
        if not file_path.startswith(os.path.abspath(PDF_DATA_DIR)):
             raise HTTPException(status_code=400, detail="Invalid filename")
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        logger.info(f"Providing download for: {file_path}")
        return FileResponse(path=file_path, filename=decoded_filename, media_type='application/pdf')
    except HTTPException as http_exc:
         raise http_exc 
    except Exception as e:
        logger.error(f"Error during file download for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during download")

# --- Health Endpoint ---
@app.get("/health")
async def health_check():
     # ... (Logic remains the same) ...
     status = "ok"; items = "N/A"
     try: 
         if embedder and index and faiss_id_to_info and metadata_store: items = index.ntotal
         else: status = "degraded - models/data not fully loaded"
     except Exception: status = "error checking status"
     return {"status": status, "model_name": MODEL_NAME, "faiss_items": items}

# --- Prometheus Instrumentation ---
from prometheus_fastapi_instrumentator import Instrumentator
try:
    Instrumentator().instrument(app).expose(app)
    logger.info("Prometheus FastAPI instrumentator attached.")
except Exception as e_instr:
     logger.error(f"Failed to attach Prometheus instrumentator: {e_instr}")