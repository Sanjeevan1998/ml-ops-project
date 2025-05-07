from transformers import AutoModel, AutoTokenizer
import os
import torch # Import torch as Legal-BERT is a PyTorch model

# --- Configuration ---
# Choose the Legal-BERT model you want to use from Hugging Face.
# This identifier is a widely used public Legal-BERT model.
model_name = "nlpaueb/legal-bert-base-uncased"

# Directory where the model and tokenizer files will be saved
save_directory = "./legal-bert-base-model"

# --- Download Logic ---
print(f"Starting download process for model: {model_name}")

# Create the save directory if it doesn't exist
# exist_ok=True prevents an error if the directory already exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory, exist_ok=True)
    print(f"Created directory: {save_directory}")
else:
    print(f"Save directory '{save_directory}' already exists.")


print(f"Attempting to download tokenizer for {model_name}...")
try:
    # Use AutoTokenizer to automatically handle different tokenizer types
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    print("Tokenizer downloaded and saved successfully.")
except Exception as e:
    print(f"Error downloading or saving tokenizer: {e}")
    # In a real script, you might want more robust error handling here.


print(f"Attempting to download model for {model_name}...")
try:
    # Use AutoModel.from_pretrained to download the base model weights.
    # For this specific model ("nlpaueb/legal-bert-base-uncased"),
    # AutoModel.from_pretrained is appropriate for getting the core BERT model.
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    print("Model downloaded and saved successfully.")
except Exception as e:
    print(f"Error downloading or saving model: {e}")
    # In a real script, you might want more robust error handling here.


print(f"\nDownload process finished.")
print(f"Model and tokenizer files should be in: {save_directory}")
print("Please verify the contents of this directory.")