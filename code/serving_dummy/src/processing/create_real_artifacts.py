# src/processing/create_real_artifacts.py
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os
import fitz # PyMuPDF
import re
import time

print("Starting REAL artifact generation...")
start_time = time.time()

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2' # Sticking with this model for now (CPU friendly)
PDF_DIR = "real_pdfs" # Directory containing your 10 PDFs
INDEX_DIR = "index" # Output directory for index/map
METADATA_DIR = "metadata" # Output directory for metadata
CHUNK_SIZE = 400  # Target size for text chunks (in tokens, approximate)
CHUNK_OVERLAP = 50 # Number of tokens to overlap between chunks

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# --- Helper Function for Basic Chunking ---
# (More sophisticated chunking can be implemented using langchain etc. later)
def simple_chunker(text, chunk_size, chunk_overlap):
    words = re.split(r'(\s+)', text) # Split by spaces, keeping separators
    tokens = [word for word in words if word.strip()] # Basic word tokenization

    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunks.append(" ".join(chunk_tokens))
        
        # Move start index for the next chunk
        step = chunk_size - chunk_overlap
        if step <= 0: # Ensure progress even with large overlap
             step = max(1, chunk_size // 2) 
        start_idx += step
        
        # Break if end_idx reached the end last time
        if end_idx == len(tokens):
            break
            
    return chunks

# --- Process PDFs ---
all_chunks_data = [] # List to hold {'source_filename': ..., 'chunk_id': ..., 'chunk_text': ...}
placeholder_metadata_store = {} # Dict to hold {filename: {metadata}}

pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files in '{PDF_DIR}'. Processing...")

for i, filename in enumerate(pdf_files):
    print(f" Processing ({i+1}/{len(pdf_files)}): {filename}...")
    file_path = os.path.join(PDF_DIR, filename)
    doc_text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                doc_text += page.get_text() + "\n" # Extract text from each page
        
        print(f"  Extracted ~{len(doc_text)} characters.")
        
        # Chunk the document text
        doc_chunks = simple_chunker(doc_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  Split into {len(doc_chunks)} chunks.")

        # Store chunk info
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunk_id = f"{filename}_chunk_{chunk_idx:04d}"
            all_chunks_data.append({
                'source_filename': filename,
                'chunk_id': chunk_id,
                'chunk_text': chunk_text
            })

        # Create placeholder metadata for this document
        placeholder_metadata_store[filename] = {
            "case_name": f"Placeholder Case Name for {filename}",
            "citation": f"Placeholder Citation {i+1}",
            "decision_date": f"2024-01-{i+1:02d}", # Dummy date
            "source_pdf_filename": filename, # Redundant but maybe useful
            "source_pdf_path": f"/app/pdf_data/{filename}", # CRITICAL: Path *inside container*
            "judge": f"Placeholder Judge {chr(65+i)}", # A, B, C...
            "outcome_summary": f"This is a placeholder summary for the document {filename}. It discusses various placeholder legal concepts.",
            "key_legal_issues": [f"Placeholder Issue {j+1} for file {i+1}" for j in range(3)]
        }
        
    except Exception as e:
        print(f"  ERROR processing {filename}: {e}. Skipping this file.")

print(f"\nTotal chunks generated from all files: {len(all_chunks_data)}")
if not all_chunks_data:
     print("ERROR: No chunks were generated. Please check PDF files and paths.")
     exit()
     
# --- Generate Embeddings ---
print(f"\nLoading embedding model: {MODEL_NAME}...")
# Use CPU for simplicity on standard VM
embedder = SentenceTransformer(MODEL_NAME, device='cpu') # Explicitly CPU

print("Generating embeddings for all chunks (this may take a while)...")
all_chunk_texts = [chunk['chunk_text'] for chunk in all_chunks_data]
embeddings = embedder.encode(all_chunk_texts, convert_to_numpy=True, show_progress_bar=True)
print(f"Generated embeddings with shape: {embeddings.shape}")
embedding_dimension = embeddings.shape[1]

# --- Build FAISS Index & Mapping ---
# Use a simple Flat L2 index for exact search - good for moderate number of vectors
index = faiss.IndexFlatL2(embedding_dimension)
print(f"Created FAISS IndexFlatL2 with dimension {embedding_dimension}")

# Create mapping from FAISS index ID (0, 1, 2...) to our chunk info
faiss_id_to_info = all_chunks_data # The list already contains the info we need per chunk

print("Adding vectors to FAISS index...")
index.add(embeddings.astype(np.float32)) # Add all embeddings in batch

print(f"Added {index.ntotal} vectors to FAISS index.")
print(f"Created FAISS mapping list with {len(faiss_id_to_info)} entries.")

# --- Save Artifacts ---
# Use new filenames to distinguish from dummy artifacts
index_path = os.path.join(INDEX_DIR, "real_index.faiss")
map_path = os.path.join(INDEX_DIR, "real_map.pkl")
metadata_path = os.path.join(METADATA_DIR, "real_metadata.pkl")

print(f"\nSaving REAL FAISS index to: {index_path}")
faiss.write_index(index, index_path)

print(f"Saving REAL FAISS map to: {map_path}")
with open(map_path, "wb") as f:
    pickle.dump(faiss_id_to_info, f)

print(f"Saving REAL metadata store to: {metadata_path}")
with open(metadata_path, "wb") as f:
    pickle.dump(placeholder_metadata_store, f)

end_time = time.time()
print(f"\nREAL artifact generation complete! Time taken: {end_time - start_time:.2f} seconds.")