from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import os

# Load the data
data_path = "/mnt/persistent_storage/processed/processed_cases.csv"
output_path = "/mnt/persistent_storage/embeddings/legal_embeddings.npy"
metadata_path = "/mnt/persistent_storage/embeddings/metadata.csv"

df = pd.read_csv(data_path)

# Load Legal-BERT
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
model.eval()

embeddings = []

for i, row in df.iterrows():
    text = row["summary"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get CLS token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    embeddings.append(cls_embedding)

# Save the embeddings
embeddings = np.array(embeddings)
np.save(output_path, embeddings)

# Save the metadata (to keep track of case names)
df[["case_name", "court", "date"]].to_csv(metadata_path, index=False)

print(f"✅ Saved {len(embeddings)} embeddings to {output_path}")
print(f"✅ Saved metadata to {metadata_path}")
