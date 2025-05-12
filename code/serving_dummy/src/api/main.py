
import os
from fastapi import FastAPI, HTTPException, Body, Query, Request, Path, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional
import json
import datetime
from pydantic import BaseModel
import time
import urllib.parse
import fitz 
import re
import io
import torch 

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
RESULTS_RETURNED_PER_QUERY_COUNT = Histogram(
    "results_returned_per_query_count",
    "Number of results (with pre-computed summaries) returned per query",
    buckets=[0, 1, 2, 3, 4, 5, 7, 10, 15, float("inf")]
)
QUERY_EMBEDDING_DURATION_SECONDS = Histogram(
    "query_embedding_duration_seconds",
    "Time taken to embed the input query (text or file chunks)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, float("inf")],
    labelnames=['query_type', 'model_type', 'device']
)
FAISS_SEARCH_DURATION_SECONDS = Histogram(
    "faiss_search_duration_seconds",
    "Time taken for a single FAISS index.search operation",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, float("inf")]
)
RESULT_AGGREGATION_DURATION_SECONDS = Histogram(
    "result_aggregation_duration_seconds",
    "Time taken to aggregate chunk results into document results",
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, float("inf")]
)
RESULT_FORMATTING_DURATION_SECONDS = Histogram(
    "result_formatting_duration_seconds",
    "Time taken by the format_final_results function",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, float("inf")]
)

MODEL_TYPE_TO_LOAD = os.environ.get("MODEL_TYPE_TO_LOAD", "ONNX").upper()
MODEL_DEVICE_PREFERENCE = os.environ.get("MODEL_DEVICE_PREFERENCE", "auto").lower() 

INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "/app/mounted_bucket_storage/faissIndex/v1/real_index.faiss")
MAP_PATH = os.environ.get("FAISS_MAP_PATH", "/app/mounted_bucket_storage/faissIndex/v1/real_map.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "/app/mounted_bucket_storage/faissIndex/v1/real_metadata.pkl")
PDF_DATA_DIR = os.environ.get("PDF_DATA_DIR", "/app/mounted_bucket_storage/LexisRaw")

PYTORCH_MODEL_NAME_OR_PATH = os.environ.get("EMBEDDING_MODEL", "/app/mounted_bucket_storage/model/Legal-BERT-finetuned")

ONNX_MODEL_PATH_ACTUAL = os.environ.get("ONNX_MODEL_PATH", "/app/optimized_models_local/legal_bert_finetuned_onnx_int8_quantized")


effective_device = "cpu" 
logger.info(f"MODEL_DEVICE_PREFERENCE from env: '{MODEL_DEVICE_PREFERENCE}'")
if MODEL_DEVICE_PREFERENCE == "cuda":
    if torch.cuda.is_available():
        effective_device = "cuda"
        logger.info("CUDA preference set and CUDA is available. Effective device: cuda.")
    else:
        logger.warning("CUDA preference set, but CUDA is NOT available. Effective device: cpu.")
        effective_device = "cpu" 
elif MODEL_DEVICE_PREFERENCE == "auto":
    if torch.cuda.is_available():
        effective_device = "cuda"
        logger.info("Device preference is 'auto' and CUDA is available. Effective device: cuda.")
    else:
        effective_device = "cpu"
        logger.info("Device preference is 'auto' and CUDA is NOT available. Effective device: cpu.")
else:
    effective_device = "cpu"
    logger.info(f"Device preference is '{MODEL_DEVICE_PREFERENCE}'. Effective device: cpu.")

class FeedbackItem(BaseModel):
    query: str
    source_pdf_filename: str
    distance: float
    feedback: str

FEEDBACK_LOG_DIR = "/app/feedback_data"
FEEDBACK_LOG_FILE = os.path.join(FEEDBACK_LOG_DIR, "feedback.jsonl")

templates = Jinja2Templates(directory="src/api/templates")

pytorch_embedder = None
onnx_model_session = None
tokenizer = None

index = None
faiss_id_to_info = []
metadata_store = {}
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

try:
    logger.info(f"Starting application setup. Attempting to load model type: {MODEL_TYPE_TO_LOAD} on effective device: {effective_device}")

    if MODEL_TYPE_TO_LOAD == "ONNX":
        if not ONNX_MODEL_PATH_ACTUAL or not os.path.isdir(ONNX_MODEL_PATH_ACTUAL):
            logger.error(f"ONNX model directory not found or not specified: {ONNX_MODEL_PATH_ACTUAL}")
            raise FileNotFoundError(f"ONNX model directory not found or not specified: {ONNX_MODEL_PATH_ACTUAL}")
        logger.info(f"Loading ONNX model from directory: {ONNX_MODEL_PATH_ACTUAL}")
        
        onnx_providers = ['CPUExecutionProvider']
        if effective_device == "cuda":
            try:
                import onnxruntime
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    logger.info("ONNX will attempt to use CUDAExecutionProvider.")
                else:
                    logger.warning("CUDAExecutionProvider not available in ONNXRuntime. Using CPU for ONNX.")
            except ImportError:
                logger.warning("ONNXRuntime not imported, cannot check for CUDAExecutionProvider. Defaulting to CPU for ONNX.")

        onnx_model_session = ORTModelForFeatureExtraction.from_pretrained(
            ONNX_MODEL_PATH_ACTUAL,
            provider=onnx_providers[0],
        )
        tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH_ACTUAL)
        logger.info(f"ONNX model and tokenizer loaded successfully from {ONNX_MODEL_PATH_ACTUAL} using provider: {onnx_model_session.device}.")

    elif MODEL_TYPE_TO_LOAD == "PYTORCH":
        logger.info(f"Loading PyTorch SentenceTransformer model: {PYTORCH_MODEL_NAME_OR_PATH} on device: {effective_device}")
        if not os.path.exists(PYTORCH_MODEL_NAME_OR_PATH):
             if PYTORCH_MODEL_NAME_OR_PATH == "all-MiniLM-L6-v2" and "EMBEDDING_MODEL" not in os.environ: 
                 logger.info("EMBEDDING_MODEL not set, but defaulting to /app/mounted_bucket_storage/model/Legal-BERT-finetuned for PyTorch if MODEL_TYPE_TO_LOAD is PYTORCH")
             elif "/" in PYTORCH_MODEL_NAME_OR_PATH: 
                 logger.warning(f"Path {PYTORCH_MODEL_NAME_OR_PATH} not found, trying to load as HuggingFace ID.")
                 pytorch_embedder = SentenceTransformer(PYTORCH_MODEL_NAME_OR_PATH.split('/')[-1], device=effective_device)
             else:
                raise FileNotFoundError(f"PyTorch model directory not found at: {PYTORCH_MODEL_NAME_OR_PATH}")
        if not pytorch_embedder: 
            pytorch_embedder = SentenceTransformer(PYTORCH_MODEL_NAME_OR_PATH, device=effective_device)
        
        if pytorch_embedder:
            tokenizer = pytorch_embedder.tokenizer 
        logger.info(f"PyTorch SentenceTransformer model loaded: {PYTORCH_MODEL_NAME_OR_PATH}")
    else:
        raise ValueError(f"Unsupported MODEL_TYPE_TO_LOAD: {MODEL_TYPE_TO_LOAD}. Choose 'PYTORCH' or 'ONNX'.")

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

app = FastAPI()

def simple_chunker(text, chunk_size, chunk_overlap):
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

def mean_pooling(model_output_last_hidden_state, attention_mask):
    token_embeddings = model_output_last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_embeddings(texts: List[str]) -> np.ndarray:
    global pytorch_embedder, onnx_model_session, tokenizer, MODEL_TYPE_TO_LOAD, effective_device
    if not texts:
        return np.array([])

    if MODEL_TYPE_TO_LOAD == "ONNX":
        if not onnx_model_session or not tokenizer:
            raise RuntimeError("ONNX model or tokenizer not loaded properly.")
        
        encoded_input = tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt' 
        )
        inputs_on_device = {k: v.to(onnx_model_session.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output_onnx = onnx_model_session(**inputs_on_device)
        
        sentence_embeddings_torch = mean_pooling(model_output_onnx.last_hidden_state, inputs_on_device['attention_mask'])
        return sentence_embeddings_torch.cpu().detach().numpy()
    
    elif MODEL_TYPE_TO_LOAD == "PYTORCH":
        if not pytorch_embedder:
            raise RuntimeError("PyTorch SentenceTransformer model not loaded properly.")
        
        return pytorch_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    else:
        raise ValueError(f"Invalid MODEL_TYPE_TO_LOAD: {MODEL_TYPE_TO_LOAD}")

def search_index_with_vector(query_vector: np.ndarray, search_k: int) -> tuple[np.ndarray, np.ndarray]:
     if not index: raise HTTPException(status_code=503, detail="FAISS Index not loaded.")
     if query_vector.ndim == 1:
         query_vector = np.expand_dims(query_vector, axis=0)
     
     effective_k = min(search_k, index.ntotal)
     if effective_k <= 0: return (np.array([]), np.array([]))
     
     t_faiss_search_start = time.time()
     distances, indices = index.search(query_vector.astype(np.float32), effective_k)
     faiss_search_duration = time.time() - t_faiss_search_start
     FAISS_SEARCH_DURATION_SECONDS.observe(faiss_search_duration)
     logger.info(f"FAISS search took {faiss_search_duration:.6f}s for k={effective_k}")
     return distances, indices

def aggregate_results_by_doc(distances: np.ndarray, indices: np.ndarray) -> Dict[str, Dict[str, Any]]:
    t_agg_start = time.time()
    doc_results = {}
    if indices.size == 0:
        RESULT_AGGREGATION_DURATION_SECONDS.observe(time.time() - t_agg_start)
        return doc_results
        
    for i, idx in enumerate(indices[0]): 
        if idx == -1 or idx >= len(faiss_id_to_info): continue
        chunk_info = faiss_id_to_info[idx]
        source_filename = chunk_info.get('source_filename')
        if not source_filename: continue
        distance = float(distances[0][i])
        similarity_score = 1.0 / (1.0 + distance) 
        if source_filename not in doc_results or similarity_score > doc_results[source_filename]['best_score']:
             doc_results[source_filename] = {
                 'best_score': similarity_score,
                 'relevant_chunk_info': chunk_info,
                 'distance': distance
             }
    agg_duration = time.time() - t_agg_start
    RESULT_AGGREGATION_DURATION_SECONDS.observe(agg_duration)
    logger.info(f"Result aggregation took {agg_duration:.6f}s, aggregated to {len(doc_results)} docs.")
    return doc_results

def format_final_results(sorted_docs: list, top_k: int) -> List[Dict[str, Any]]:
     t_format_start = time.time()
     formatted_results = []
     docs_to_return = sorted_docs[:top_k]
     num_docs_returned = len(docs_to_return)

     logger.info(f"Preparing to return {num_docs_returned} results with pre-computed summaries.")
     RESULTS_RETURNED_PER_QUERY_COUNT.observe(num_docs_returned)

     for filename, data in docs_to_return:
         doc_metadata = metadata_store.get(filename, {})
         chunk_info = data['relevant_chunk_info']
         formatted_results.append({
             "case_name": doc_metadata.get("case_name", "N/A"),
             "citation": doc_metadata.get("citation", "N/A"),
             "decision_date": doc_metadata.get("decision_date", "N/A"),
             "judge": doc_metadata.get("judge", "N/A"),
             "outcome_summary": doc_metadata.get("outcome_summary", "N/A - Pre-computed summary"),
             "key_legal_issues": doc_metadata.get("key_legal_issues", []),
             "source_pdf_filename": filename,
             "relevant_chunk_text": chunk_info.get('chunk_text', 'N/A'),
             "similarity_score": data['best_score'],
             "distance": data['distance']
         })
     
     format_duration = time.time() - t_format_start
     RESULT_FORMATTING_DURATION_SECONDS.observe(format_duration)
     logger.info(f"Result formatting took {format_duration:.6f}s for {num_docs_returned} docs.")
     return formatted_results

async def perform_combined_search(query: Optional[str], query_file: Optional[UploadFile], top_k: int) -> List[Dict[str, Any]]:
    model_ready = False
    if MODEL_TYPE_TO_LOAD == "PYTORCH" and pytorch_embedder and tokenizer: model_ready = True
    elif MODEL_TYPE_TO_LOAD == "ONNX" and onnx_model_session and tokenizer: model_ready = True

    if not model_ready:
        logger.error(f"Search attempted before {MODEL_TYPE_TO_LOAD} model/tokenizer loaded.")
        raise HTTPException(status_code=503, detail=f"Service ({MODEL_TYPE_TO_LOAD} Model) not fully initialized.")
    if not index or not faiss_id_to_info or not metadata_store:
         logger.error("Search attempted before FAISS/metadata loaded.")
         raise HTTPException(status_code=503, detail="Service (Data) not fully initialized.")
    if not query and not query_file:
         raise HTTPException(status_code=400, detail="Please provide a text query or upload a file.")

    t_start_total_search_logic = time.time()
    all_doc_results_detailed = {}
    
    current_device_label = effective_device 

    if query:
        logger.info(f"Processing text query: '{query[:100]}...' using {MODEL_TYPE_TO_LOAD} on {effective_device}")
        t_embed_start_mono = time.monotonic()
        query_embeddings_np = get_embeddings([query])
        embed_duration_mono = time.monotonic() - t_embed_start_mono
        
        QUERY_EMBEDDING_DURATION_SECONDS.labels(query_type='text', model_type=MODEL_TYPE_TO_LOAD.lower(), device=current_device_label).observe(embed_duration_mono)
        logger.info(f"Text query embedding took {embed_duration_mono:.6f}s.")
        
        if query_embeddings_np.size > 0:
            search_k_text = min(max(top_k * 2, 10), index.ntotal) 
            distances, indices = search_index_with_vector(query_embeddings_np, search_k_text)
            doc_results_text = aggregate_results_by_doc(distances, indices)
            for filename, data in doc_results_text.items():
                if filename not in all_doc_results_detailed or data['best_score'] > all_doc_results_detailed[filename]['best_score']:
                    all_doc_results_detailed[filename] = data
            if indices.size > 0 and indices[0][0] != -1:
                 SEARCH_CLOSEST_DISTANCE.observe(float(distances[0][0]))
        else:
            logger.warning("Text query resulted in empty embeddings.")

    file_chunks_to_embed = []
    if query_file and query_file.filename:
         logger.info(f"Processing uploaded file: {query_file.filename}")
         if not query_file.filename.lower().endswith(".pdf"):
               logger.warning(f"Invalid file type uploaded: {query_file.filename}")
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
                       file_chunks_to_embed = simple_chunker(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
                       logger.info(f"Split uploaded file into {len(file_chunks_to_embed)} chunks.")
             except Exception as e:
                  logger.error(f"Error processing uploaded file {query_file.filename}: {e}", exc_info=True)

    if file_chunks_to_embed:
        logger.info(f"Embedding {len(file_chunks_to_embed)} file chunks using {MODEL_TYPE_TO_LOAD} on {effective_device}")
        t_embed_file_start_mono = time.monotonic()
        chunk_embeddings_np = get_embeddings(file_chunks_to_embed)
        embed_file_duration_mono = time.monotonic() - t_embed_file_start_mono
        
        QUERY_EMBEDDING_DURATION_SECONDS.labels(query_type='file', model_type=MODEL_TYPE_TO_LOAD.lower(), device=current_device_label).observe(embed_file_duration_mono)
        logger.info(f"File query embedding ({len(file_chunks_to_embed)} chunks) took {embed_file_duration_mono:.6f}s.")

        if chunk_embeddings_np.size > 0:
            search_k_per_chunk = min(max(top_k // len(file_chunks_to_embed) +1 if len(file_chunks_to_embed) > 0 else top_k, 3), index.ntotal) 
            
            for vec_idx, vec_np in enumerate(chunk_embeddings_np):
                logger.info(f"Searching for file chunk {vec_idx+1}/{len(file_chunks_to_embed)}")
                distances_f, indices_f = search_index_with_vector(vec_np, search_k_per_chunk)
                doc_results_chunk = aggregate_results_by_doc(distances_f, indices_f)
                for filename, data in doc_results_chunk.items():
                     if filename not in all_doc_results_detailed or data['best_score'] > all_doc_results_detailed[filename]['best_score']:
                          all_doc_results_detailed[filename] = data
        else:
            logger.warning("File processing resulted in no embeddings for its chunks.")

    if not all_doc_results_detailed:
         logger.info("No results found from either text or file query.")
         RESULTS_RETURNED_PER_QUERY_COUNT.observe(0)
         return []

    sorted_detailed_docs = sorted(all_doc_results_detailed.items(), key=lambda item: item[1]['best_score'], reverse=True)
    final_results_with_metadata = format_final_results(sorted_detailed_docs, top_k)
    
    total_search_logic_duration = time.time() - t_start_total_search_logic
    logger.info(f"Core search logic for {MODEL_TYPE_TO_LOAD} on {effective_device} finished in {total_search_logic_duration:.4f}s. Returning {len(final_results_with_metadata)} documents.")
    return final_results_with_metadata

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search_combined", response_class=HTMLResponse)
async def search_combined_post_html(
    request: Request,
    top_k: int = Form(default=5),
    query: Optional[str] = Form(None),
    query_file: Optional[UploadFile] = File(None)
):
    display_query_context = query if query else (query_file.filename if query_file and query_file.filename else "N/A")
    try:
        search_results = await perform_combined_search(query=query, query_file=query_file, top_k=top_k)
        return templates.TemplateResponse("results.html", {"request": request, "results": search_results, "query": display_query_context})
    except HTTPException as http_exc:
         return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": http_exc.detail, "query": display_query_context})
    except Exception as e:
        logger.error(f"Error in POST /search_combined endpoint: {e}", exc_info=True)
        return templates.TemplateResponse("results.html", {"request": request, "results": None, "error": f"Internal server error during combined search with {MODEL_TYPE_TO_LOAD} on {effective_device}.", "query": display_query_context})

@app.post("/search") 
async def search_documents_post_api(query_text: Optional[str] = Body(None, embed=True), top_k: int = Body(default=5, embed=True)):
    if not query_text:
        raise HTTPException(status_code=400, detail="Please provide 'query_text'.")
    try:
         search_results_detailed = await perform_combined_search(query=query_text, query_file=None, top_k=top_k)
         return {"results": search_results_detailed, "model_used": MODEL_TYPE_TO_LOAD, "device_used": effective_device}
    except HTTPException as http_exc: 
         raise http_exc
    except Exception as e:
        logger.error(f"Error in POST /search API endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during API search with {MODEL_TYPE_TO_LOAD} on {effective_device}.")

@app.post("/log_feedback")
async def log_feedback(payload: FeedbackItem):
    try:
        feedback_type = payload.feedback.lower()
        FEEDBACK_COUNTER.labels(feedback_type=feedback_type).inc()
        log_entry = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "query": payload.query,
            "source_pdf_filename": payload.source_pdf_filename,
            "distance": payload.distance,
            "feedback": feedback_type,
            "model_type_active_at_feedback": MODEL_TYPE_TO_LOAD,
            "device_active_at_feedback": effective_device
        }
        with open(FEEDBACK_LOG_FILE, "a") as f: f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Feedback logged: {log_entry}")
        return {"status": "success", "message": "Feedback logged"}
    except Exception as e:
        logger.error(f"Error logging feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to log feedback")

@app.get("/download/{filename:path}")
async def download_pdf(filename: str = Path(..., description="The name of the PDF file to download")):
    global PDF_DATA_DIR  
    if not PDF_DATA_DIR:
        logger.error("PDF_DATA_DIR is not configured properly.")
        raise HTTPException(status_code=500, detail="Server configuration error: PDF directory not set.")
    try:
        decoded_filename = urllib.parse.unquote(filename)
        if ".." in decoded_filename or decoded_filename.startswith("/") or "\\" in decoded_filename:
             logger.warning(f"Attempted path traversal in download: {filename}")
             raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = os.path.abspath(os.path.join(PDF_DATA_DIR, decoded_filename))
        
        if not file_path.startswith(os.path.abspath(PDF_DATA_DIR)):
             logger.warning(f"Attempted to access file outside PDF_DATA_DIR: {file_path}")
             raise HTTPException(status_code=400, detail="Invalid filename (access denied)")
            
        if not os.path.isfile(file_path):
            logger.warning(f"Download request for non-existent file: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"Providing download for: {file_path}")
        return FileResponse(path=file_path, filename=decoded_filename, media_type='application/pdf')
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"Error during file download for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during download")

@app.get("/health")
async def health_check():
     global effective_device, MODEL_TYPE_TO_LOAD, ONNX_MODEL_PATH_ACTUAL, PYTORCH_MODEL_NAME_OR_PATH, index, onnx_model_session, pytorch_embedder, tokenizer
     status = "ok"
     items_in_faiss = "N/A"
     current_model_name_reported = "N/A"
     model_loaded_check = False

     try:
         if MODEL_TYPE_TO_LOAD == "ONNX":
             if onnx_model_session and tokenizer:
                 current_model_name_reported = ONNX_MODEL_PATH_ACTUAL
                 model_loaded_check = True
             else:
                 status = "degraded - ONNX model/tokenizer not fully loaded"
         elif MODEL_TYPE_TO_LOAD == "PYTORCH":
             if pytorch_embedder and tokenizer:
                 current_model_name_reported = PYTORCH_MODEL_NAME_OR_PATH
                 model_loaded_check = True
             else:
                 status = "degraded - PyTorch model/tokenizer not fully loaded"
         else:
            status = f"error - Unknown MODEL_TYPE_TO_LOAD: {MODEL_TYPE_TO_LOAD}"

         if not model_loaded_check and status == "ok":
             status = "degraded - Model not loaded (no specific type error, but not ready)."

         if index:
             items_in_faiss = index.ntotal
         else:
             status = "degraded - FAISS index not loaded"
             
     except Exception as e:
         status = f"error checking status: {str(e)}"
         logger.error(f"Health check error: {e}", exc_info=True)

     return {
         "status": status,
         "active_model_type": MODEL_TYPE_TO_LOAD,
         "active_model_path_or_name": current_model_name_reported,
         "preferred_device_env": MODEL_DEVICE_PREFERENCE,
         "effective_device_in_use": effective_device, 
         "torch_cuda_available": torch.cuda.is_available(), 
         "faiss_items": items_in_faiss
    }

from prometheus_fastapi_instrumentator import Instrumentator
try:
    Instrumentator(
        should_instrument_requests_inprogress=True,
        inprogress_labels=True,
    ).instrument(app).expose(app)
    logger.info("Prometheus FastAPI instrumentator attached.")
except Exception as e_instr:
     # Your original log:
     logger.error(f"Failed to attach Prometheus instrumentator: {e_instr}")