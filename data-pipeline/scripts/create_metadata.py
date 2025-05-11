import pandas as pd
import os

PROCESSED_FILE = "/data/processed/processed_cases.csv"
METADATA_FILE = "/data/embeddings/metadata.csv"

# Create directories if they don't exist
os.makedirs("/data/embeddings/", exist_ok=True)

# Check if the processed file exists
if not os.path.exists(PROCESSED_FILE):
    print(f"❌ Error: {PROCESSED_FILE} not found. Make sure to run process_pdfs.py first.")
    exit(1)

# Load the processed cases
df = pd.read_csv(PROCESSED_FILE)

# Check if the dataframe is empty
if df.empty:
    print(f"❌ Error: {PROCESSED_FILE} is empty. Cannot create metadata.")
    exit(1)

# Extract only metadata columns
metadata_df = df[["case_name", "court", "date"]]

# Save the metadata file
metadata_df.to_csv(METADATA_FILE, index=False)

print(f"✅ Saved metadata to {METADATA_FILE}")
