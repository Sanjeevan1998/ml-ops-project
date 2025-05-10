from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import os

# File paths
input_file = "/data-pipeline/processed/processed_cases.csv"
embedding_file = "/data-pipeline/embeddings/legal_embeddings.npy"
metadata_file = "/data-pipeline/embeddings/metadata.csv"

# Create directories if they don't exist
os.makedirs("/data-pipeline/embeddings/", exist_ok=True)

# Load the data
df = pd.read_csv(input_file)

# Load the Legal-BERT model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
model.eval()

embeddings = []
metadata = []

print(f"📚 Processing {len(df)} cases...")

for index, row in df.iterrows():
    case_name = row["case_name"]
    summary = row["summary"]

    # Tokenize and embed
    inputs = tokenizer(summary, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Use the [CLS] token as the sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Store the embedding
    embeddings.append(cls_embedding)
    metadata.append([case_name, row["court"], row["date"]])

    if (index + 1) % 100 == 0:
        print(f"✅ Processed {index+1} cases...")

# Convert to NumPy array
embeddings = np.array(embeddings)

# Save embeddings
np.save(embedding_file, embeddings)
print(f"✅ Saved embeddings to {embedding_file}")

# Save metadata
pd.DataFrame(metadata, columns=["case_name", "court", "date"]).to_csv(metadata_file, index=False)
print(f"✅ Saved metadata to {metadata_file}")

