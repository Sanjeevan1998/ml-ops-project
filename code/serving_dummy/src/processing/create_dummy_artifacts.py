import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

print("Starting dummy artifact generation...")

MODEL_NAME = 'all-MiniLM-L6-v2'

INDEX_DIR = "index"
METADATA_DIR = "metadata"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


dummy_texts = [
    "This is the first dummy document about contract law.",
    "The second document discusses tort law and negligence.",
    "A third piece of text focusing on intellectual property rights.",
    "Another contract law example for testing similarity.",
    "Finally, a note on civil procedure.",
]

dummy_metadata_list = [
    {"case_name": "Case A", "citation": "1 F. Supp. 1", "decision_date": "2023-01-01", "source_pdf_filename": "dummy_doc_1.txt", "chunk_id": "chunk_0"},
    {"case_name": "Case B", "citation": "2 F. Supp. 2", "decision_date": "2023-02-02", "source_pdf_filename": "dummy_doc_2.txt", "chunk_id": "chunk_1"},
    {"case_name": "Case C", "citation": "3 F. Supp. 3", "decision_date": "2023-03-03", "source_pdf_filename": "dummy_doc_3.txt", "chunk_id": "chunk_2"},
    {"case_name": "Case D", "citation": "4 F. Supp. 4", "decision_date": "2023-04-04", "source_pdf_filename": "dummy_doc_4.txt", "chunk_id": "chunk_3"},
    {"case_name": "Case E", "citation": "5 F. Supp. 5", "decision_date": "2023-05-05", "source_pdf_filename": "dummy_doc_5.txt", "chunk_id": "chunk_4"},
]

dummy_metadata_store = {meta["source_pdf_filename"]: meta for meta in dummy_metadata_list}


print(f"Loaded {len(dummy_texts)} dummy documents.")

print(f"Loading embedding model: {MODEL_NAME}...")

embedder = SentenceTransformer(MODEL_NAME, device='cpu')
print("Generating embeddings...")
embeddings = embedder.encode(dummy_texts, convert_to_numpy=True)
print(f"Generated embeddings with shape: {embeddings.shape}")
embedding_dimension = embeddings.shape[1]


index = faiss.IndexFlatL2(embedding_dimension)
print(f"Created FAISS IndexFlatL2 with dimension {embedding_dimension}")


faiss_id_to_info = []
for i, meta in enumerate(dummy_metadata_list):
     faiss_id_to_info.append({
         'source_filename': meta['source_pdf_filename'],
         'chunk_id': meta['chunk_id'] 
     })

     index.add(np.array([embeddings[i]])) 

print(f"Added {index.ntotal} vectors to FAISS index.")
print(f"Created FAISS mapping list with {len(faiss_id_to_info)} entries.")


index_path = os.path.join(INDEX_DIR, "dummy_index.faiss")
map_path = os.path.join(INDEX_DIR, "dummy_map.pkl")
metadata_path = os.path.join(METADATA_DIR, "dummy_metadata.pkl")

print(f"Saving FAISS index to: {index_path}")
faiss.write_index(index, index_path)

print(f"Saving FAISS map to: {map_path}")
with open(map_path, "wb") as f:
    pickle.dump(faiss_id_to_info, f)

print(f"Saving metadata store to: {metadata_path}")
with open(metadata_path, "wb") as f:
    pickle.dump(dummy_metadata_store, f)

print("Dummy artifact generation complete!")