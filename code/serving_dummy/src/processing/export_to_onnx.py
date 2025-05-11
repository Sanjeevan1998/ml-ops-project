# src/processing/export_to_onnx.py
import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer # Still needed to get the exact tokenizer config initially
import time

# --- Configuration ---
# Path to your original fine-tuned SentenceTransformer model on the VM
ORIGINAL_MODEL_PATH = "/mnt/object-store-persist-group36/model/Legal-BERT-finetuned"

# Directory where the ONNX model will be saved locally on the VM
# You can choose any temporary path on the VM's local disk.
# Let's create it relative to where this script might be run from, or an absolute path.
# For simplicity, using a path in /tmp or a local project path is fine.
# We will create a subdirectory for this specific ONNX model.
ONNX_EXPORT_SUBDIR = "legal_bert_finetuned_onnx"
# Create a path in the user's home directory for simplicity, or /tmp
# LOCAL_OPTIMIZED_MODELS_DIR = os.path.expanduser("~/optimized_models")
LOCAL_OPTIMIZED_MODELS_DIR = "/tmp/optimized_models" # Easier for quick tests

ONNX_MODEL_SAVE_PATH = os.path.join(LOCAL_OPTIMIZED_MODELS_DIR, ONNX_EXPORT_SUBDIR)

# Make sure the save directory exists
os.makedirs(ONNX_MODEL_SAVE_PATH, exist_ok=True)

print(f"Original PyTorch Model Path: {ORIGINAL_MODEL_PATH}")
print(f"ONNX Model Save Path: {ONNX_MODEL_SAVE_PATH}")

def main():
    print("Starting ONNX export process...")
    start_time = time.time()

    # 1. Load the original SentenceTransformer to easily get its underlying Hugging Face model name or path
    # This step helps ensure we use the exact architecture and tokenizer the ST model was built upon.
    # If ORIGINAL_MODEL_PATH directly contains a HuggingFace model (pytorch_model.bin, config.json),
    # Optimum might be able to load it directly.
    # However, SentenceTransformer objects wrap a HuggingFace model.
    # Let's try loading with Optimum directly, assuming ORIGINAL_MODEL_PATH has the HF structure.
    # If not, we might need to load with SentenceTransformer first to get the HF model name.

    try:
        # We need a tokenizer from the original model for Optimum
        print(f"Loading tokenizer from: {ORIGINAL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
        print("Tokenizer loaded successfully.")

        # Export the model to ONNX using Optimum
        # ORTModelForFeatureExtraction is suitable for sentence embedding models
        # as they are often used as feature extractors.
        print(f"Exporting model from '{ORIGINAL_MODEL_PATH}' to ONNX format...")
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            ORIGINAL_MODEL_PATH,
            export=True, # This flag tells Optimum to perform the export
            # config=model.auto_model.config # If you had loaded ST model first
        )
        print("Model export to ONNX initiated by Optimum.")

        # Save the ONNX model and its tokenizer
        print(f"Saving ONNX model and tokenizer to: {ONNX_MODEL_SAVE_PATH}")
        ort_model.save_pretrained(ONNX_MODEL_SAVE_PATH)
        tokenizer.save_pretrained(ONNX_MODEL_SAVE_PATH) # Save tokenizer alongside ONNX model

        end_time = time.time()
        print(f"ONNX model and tokenizer saved successfully to {ONNX_MODEL_SAVE_PATH}")
        print(f"Conversion to ONNX took {end_time - start_time:.2f} seconds.")

        print("\nContents of the ONNX model directory:")
        for item in os.listdir(ONNX_MODEL_SAVE_PATH):
            print(f"- {item}")

    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")
        print("Troubleshooting tips:")
        print("- Ensure the ORIGINAL_MODEL_PATH is correct and points to a valid Hugging Face model directory.")
        print("- Ensure 'optimum[onnxruntime]' and 'onnx' are installed.")
        print("- If the model is a SentenceTransformer, ensure the path contains the underlying transformer files.")
        

if __name__ == "__main__":
    main()