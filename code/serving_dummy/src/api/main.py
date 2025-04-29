# Example structure for src/api/main.py
from fastapi import FastAPI, HTTPException, Body, Request
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
from prometheus_fastapi_instrumentator import Instrumentator # For Unit 6/7

# --- Configuration ---
INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/index/legal_index.faiss")
MAP_PATH = os.environ.get("FAISS_MAP_PATH", "/app/index/faiss_map.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/metadata/metadata_store.pkl")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "nlpaueb/legal-bert-base-uncased") # Example
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "cpu") # Set to 'cuda' for GPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        metadata_store = pickle.load(f) # Assuming dict: filename -> metadata
    logger.info("Metadata store loaded.")

except Exception as e:
    logger.error(f"FATAL: Error loading models/data: {e}", exc_info=True)
    # Depending on setup, you might want the app to exit or enter a degraded state
    raise RuntimeError(f"Failed to initialize application: {e}")

app = FastAPI()

# --- Unit 6/7: Instrument for Prometheus ---
Instrumentator().instrument(app).expose(app)
logger.info("Prometheus instrumentator attached.")

# --- API Endpoints ---
@app.post("/search")
async def search_documents(query: str = Body(...), top_k: int = Body(default=5)):
    try:
        logger.info(f"Received search request: top_k={top_k}, query='{query[:100]}...'") # Log truncated query

        # 1. Embed Query
        query_vector = embedder.encode([query], convert_to_numpy=True, device=MODEL_DEVICE)

        # 2. Search FAISS
        # FAISS expects float32. Ensure query_vector matches index dtype if needed.
        distances, indices = index.search(query_vector.astype(np.float32), top_k)

        # 3. Retrieve & Enrich Results
        results = []
        processed_filenames = set() # To deduplicate results by source document

        for i, idx in enumerate(indices[0]):
            if idx == -1: # Should not happen with HNSW unless k > ntotal, but good check
                continue

            # Ensure index is within bounds of the map
            if idx >= len(faiss_id_to_info):
                logger.warning(f"FAISS index {idx} out of bounds for map length {len(faiss_id_to_info)}")
                continue

            chunk_info = faiss_id_to_info[idx]
            source_filename = chunk_info.get('source_filename')
            chunk_id = chunk_info.get('chunk_id') # You'll need the chunk text too eventually

            if not source_filename:
                 logger.warning(f"Missing source_filename for FAISS index {idx}")
                 continue

            # Optional: Deduplicate based on source document
            # if source_filename in processed_filenames:
            #    continue
            # processed_filenames.add(source_filename)

            doc_metadata = metadata_store.get(source_filename, {})
            distance = distances[0][i]
            # You might need to retrieve the actual chunk text here if needed in response

            results.append({
                "case_name": doc_metadata.get("case_name", "N/A"),
                "citation": doc_metadata.get("citation", "N/A"),
                "decision_date": doc_metadata.get("decision_date", "N/A"),
                "outcome_summary": doc_metadata.get("outcome_summary", "N/A"),
                "source_pdf_filename": source_filename,
                "chunk_id": chunk_id,
                # "relevant_chunk_text": "...", # Retrieve this if needed
                "distance": float(distance), # Lower is better for L2/IP
                # "similarity_score": calculate_similarity(distance), # Implement if needed
            })

        logger.info(f"Returning {len(results)} results for query '{query[:100]}...'")
        return {"results": results}

    except Exception as e:
        logger.error(f"Error processing search request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")


@app.post("/feedback")
async def log_feedback(request: Request, feedback_data: dict = Body(...)):
    # Example feedback_data: {"query": "...", "source_pdf_filename": "...", "chunk_id": "...", "is_relevant": true/false}
    client_host = request.client.host
    logger.info(f"Received feedback from {client_host}: {feedback_data}")
    # --- Unit 7: Store Feedback ---
    # Implement logic to save this feedback persistently (e.g., write to a log file,
    # send to a MinIO bucket, insert into a database). This is crucial for Phase 3/4.
    # Example: append to a JSONL file
    # with open("/app/feedback_log/feedback.jsonl", "a") as f:
    #     json.dump({"timestamp": datetime.utcnow().isoformat(), "client": client_host, **feedback_data}, f)
    #     f.write("\n")
    return {"status": "feedback received"}

@app.get("/health")
async def health_check():
    # Basic health check
    return {"status": "ok", "faiss_items": index.ntotal}

# Add other endpoints as needed (e.g., for metadata lookup)