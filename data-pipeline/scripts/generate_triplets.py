import faiss
import numpy as np
import pandas as pd
import random

# File paths
embedding_file = "/data/embeddings/legal_embeddings.npy"
metadata_file = "/data/embeddings/metadata.csv"
triplet_output_file = "/data/scripts/triplets.csv"

# Load embeddings and metadata
embeddings = np.load(embedding_file)
metadata = pd.read_csv(metadata_file)

# Load the FAISS index
index = faiss.IndexFlatL2(768)  # 768 is the embedding dimension for Legal-BERT
index.add(embeddings)

# Generate triplets
triplets = []

for i in range(len(embeddings)):
    # Anchor
    anchor_text = metadata.iloc[i]["case_name"]
    anchor_vector = embeddings[i].reshape(1, -1)

    # Find the most similar cases (excluding the anchor itself)
    distances, indices = index.search(anchor_vector, k=6)  # top 5 similar + self

    # Positive is the second closest (excluding self)
    positive_index = indices[0][1]
    positive_text = metadata.iloc[positive_index]["case_name"]

    # Select a random negative (ensure it's not in the top 5)
    all_indices = set(range(len(embeddings)))
    negative_index = random.choice(list(all_indices - set(indices[0])))
    negative_text = metadata.iloc[negative_index]["case_name"]

    # Append the triplet
    triplets.append([anchor_text, positive_text, negative_text])

    if (i + 1) % 100 == 0:
        print(f"✅ Generated {i+1} triplets...")

# Save to CSV
triplet_df = pd.DataFrame(triplets, columns=["anchor", "positive", "negative"])
triplet_df.to_csv(triplet_output_file, index=False)

print(f"\n✅ Saved {len(triplets)} triplets to {triplet_output_file}")
