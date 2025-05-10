import pandas as pd
import os

PROCESSED_FILE = "/data/processed/processed_cases.csv"
METADATA_FILE = "/data/embeddings/metadata.csv"

# Create directories if they don't exist
os.makedirs("/data/embeddings/", exist_ok=True)

# Load the processed cases
df = pd.read_csv(PROCESSED_FILE)

# Extract only metadata columns
metadata_df = df[["case_name", "court", "date"]]

# Save the metadata file
metadata_df.to_csv(METADATA_FILE, index=False)

print(f"✅ Saved metadata to {METADATA_FILE}")
