# src/processing/create_artifacts_from_object_storage.py
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import fitz  # PyMuPDF
import re
import time
import shutil # For creating/cleaning output directory
import torch # Import torch to check for CUDA availability

print("Starting Artifact Generation from Object Storage Mount...")
start_time = time.time()

# --- Configuration ---
# Hardcoded paths for model and source PDFs on the VM (via rclone mount)
MODEL_PATH_ON_VM = '/mnt/object-store-persist-group36/model/Legal-BERT-finetuned'
PDF_SOURCE_DIR_ON_VM = '/mnt/object-store-persist-group36/LexisRaw'

# Output directory for generated artifacts on the VM (local staging)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))
LOCAL_OUTPUT_BASE_DIR_NAME = 'generated_artifacts_from_os'
OUTPUT_ARTIFACT_SUBDIR_NAME = 'faissIndex/v1'
OUTPUT_ARTIFACT_DIR_ON_VM = os.path.join(PROJECT_ROOT_DIR, LOCAL_OUTPUT_BASE_DIR_NAME, OUTPUT_ARTIFACT_SUBDIR_NAME)

MAX_FILES_TO_PROCESS = 10  # For testing; change to a larger number or None for full processing

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# --- Determine Device (GPU or CPU) ---
if torch.cuda.is_available():
    device = 'cuda'
    print(f"CUDA (GPU) is available. Using device: {device}")
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}") # Prints the name of the first GPU
    except Exception as e:
        print(f"Could not get GPU name: {e}")
else:
    device = 'cpu'
    print("CUDA (GPU) is not available or PyTorch CUDA version not installed. Using device: cpu")


# --- Helper Function for Basic Chunking ---
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
        if step <= 0:
             step = max(1, chunk_size // 2)
        start_idx += step
        if end_idx == len(tokens):
            break
    return chunks

# --- Main Script Logic ---
print(f"Model path: {MODEL_PATH_ON_VM}")
print(f"PDF source directory: {PDF_SOURCE_DIR_ON_VM}")
print(f"Output artifact directory: {OUTPUT_ARTIFACT_DIR_ON_VM}")
if MAX_FILES_TO_PROCESS is not None:
    print(f"Processing a maximum of {MAX_FILES_TO_PROCESS} PDF files.")
else:
    print("Processing all PDF files found.")

if os.path.exists(OUTPUT_ARTIFACT_DIR_ON_VM):
    print(f"Cleaning existing output directory: {OUTPUT_ARTIFACT_DIR_ON_VM}")
    shutil.rmtree(OUTPUT_ARTIFACT_DIR_ON_VM)
os.makedirs(OUTPUT_ARTIFACT_DIR_ON_VM, exist_ok=True)
print(f"Ensured output directory exists: {OUTPUT_ARTIFACT_DIR_ON_VM}")

all_pdf_files_in_source = []
try:
    if not os.path.isdir(PDF_SOURCE_DIR_ON_VM):
        print(f"ERROR: PDF source directory does not exist or is not accessible: {PDF_SOURCE_DIR_ON_VM}")
        exit()
    all_pdf_files_in_source = sorted([f for f in os.listdir(PDF_SOURCE_DIR_ON_VM) if f.lower().endswith(".pdf")])
except Exception as e:
    print(f"ERROR: Could not list files in PDF source directory '{PDF_SOURCE_DIR_ON_VM}': {e}")
    exit()

if not all_pdf_files_in_source:
    print(f"No PDF files found in '{PDF_SOURCE_DIR_ON_VM}'. Exiting.")
    exit()

if MAX_FILES_TO_PROCESS is not None and MAX_FILES_TO_PROCESS < len(all_pdf_files_in_source):
    pdf_files_to_process = all_pdf_files_in_source[:MAX_FILES_TO_PROCESS]
    print(f"Selected {len(pdf_files_to_process)} files out of {len(all_pdf_files_in_source)} for processing.")
else:
    pdf_files_to_process = all_pdf_files_in_source
    print(f"Selected all {len(pdf_files_to_process)} files for processing.")

all_chunks_data = []
placeholder_metadata_store = {}

print(f"\nProcessing {len(pdf_files_to_process)} PDF files...")
for i, filename in enumerate(pdf_files_to_process):
    print(f" Processing ({i+1}/{len(pdf_files_to_process)}): {filename}...")
    file_path = os.path.join(PDF_SOURCE_DIR_ON_VM, filename)
    doc_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                doc_text += page.get_text() + "\n"
        if not doc_text.strip():
            print(f"  WARNING: No text extracted from {filename}. Skipping.")
            continue
        print(f"  Extracted ~{len(doc_text)} characters.")
        doc_chunks = simple_chunker(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  Split into {len(doc_chunks)} chunks.")
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunk_id = f"{filename}_chunk_{chunk_idx:04d}"
            all_chunks_data.append({
                'source_filename': filename,
                'chunk_id': chunk_id,
                'chunk_text': chunk_text
            })
        placeholder_metadata_store[filename] = {
            "case_name": f"Case Name for {os.path.splitext(filename)[0]}",
            "citation": f"Citation {i+1}",
            "decision_date": f"2024-01-{i+1:02d}",
            "source_pdf_filename": filename,
            "source_pdf_path_in_api": filename,
            "judge": f"Judge {chr(65+(i%26))}",
            "outcome_summary": f"This is a placeholder summary for the document {filename}.",
            "key_legal_issues": [f"Legal Issue {j+1} for file {i+1}" for j in range(np.random.randint(2,5))]
        }
    except Exception as e:
        print(f"  ERROR processing {filename}: {e}. Skipping this file.")

print(f"\nTotal chunks generated from all processed files: {len(all_chunks_data)}")
if not all_chunks_data:
     print("ERROR: No chunks were generated. Please check PDF files and paths, or increase MAX_FILES_TO_PROCESS.")
     exit()

print(f"\nLoading embedding model: {MODEL_PATH_ON_VM} using device: {device}...")
try:
    # Pass the determined device to SentenceTransformer
    embedder = SentenceTransformer(MODEL_PATH_ON_VM, device=device)
except Exception as e:
    print(f"ERROR: Could not load SentenceTransformer model from '{MODEL_PATH_ON_VM}': {e}")
    print("Ensure the path is correct and the directory contains a valid Hugging Face model structure.")
    exit()

print("Generating embeddings for all chunks (this may take a while)...")
all_chunk_texts = [chunk['chunk_text'] for chunk in all_chunks_data]
# The encode method will use the device specified during SentenceTransformer initialization
embeddings = embedder.encode(all_chunk_texts, convert_to_numpy=True, show_progress_bar=True)
print(f"Generated embeddings with shape: {embeddings.shape}")
embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(embedding_dimension)
print(f"Created FAISS IndexFlatL2 with dimension {embedding_dimension}")
faiss_id_to_info = all_chunks_data
print("Adding vectors to FAISS index...")
index.add(embeddings.astype(np.float32))
print(f"Added {index.ntotal} vectors to FAISS index.")
print(f"FAISS map (faiss_id_to_info) created with {len(faiss_id_to_info)} entries.")

index_filename = "real_index.faiss"
map_filename = "real_map.pkl"
metadata_filename = "real_metadata.pkl"

index_path_output = os.path.join(OUTPUT_ARTIFACT_DIR_ON_VM, index_filename)
map_path_output = os.path.join(OUTPUT_ARTIFACT_DIR_ON_VM, map_filename)
metadata_path_output = os.path.join(OUTPUT_ARTIFACT_DIR_ON_VM, metadata_filename)

print(f"\nSaving FAISS index to: {index_path_output}")
faiss.write_index(index, index_path_output)
print(f"Saving FAISS map to: {map_path_output}")
with open(map_path_output, "wb") as f:
    pickle.dump(faiss_id_to_info, f)
print(f"Saving metadata store to: {metadata_path_output}")
with open(metadata_path_output, "wb") as f:
    pickle.dump(placeholder_metadata_store, f)

end_time = time.time()
print(f"\nArtifact generation complete! Time taken: {end_time - start_time:.2f} seconds.")
print(f"Artifacts saved in: {OUTPUT_ARTIFACT_DIR_ON_VM}")
print("You can now manually upload these files to your persistent object storage if needed.")