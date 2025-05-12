import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer # Still needed to get the exact tokenizer config initially
import time


ORIGINAL_MODEL_PATH = "/mnt/object-store-persist-group36/model/Legal-BERT-finetuned"



ONNX_EXPORT_SUBDIR = "legal_bert_finetuned_onnx"


LOCAL_OPTIMIZED_MODELS_DIR = "/tmp/optimized_models" 

ONNX_MODEL_SAVE_PATH = os.path.join(LOCAL_OPTIMIZED_MODELS_DIR, ONNX_EXPORT_SUBDIR)


os.makedirs(ONNX_MODEL_SAVE_PATH, exist_ok=True)

print(f"Original PyTorch Model Path: {ORIGINAL_MODEL_PATH}")
print(f"ONNX Model Save Path: {ONNX_MODEL_SAVE_PATH}")

def main():
    print("Starting ONNX export process...")
    start_time = time.time()



    try:
        print(f"Loading tokenizer from: {ORIGINAL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
        print("Tokenizer loaded successfully.")


        
        print(f"Exporting model from '{ORIGINAL_MODEL_PATH}' to ONNX format...")
        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            ORIGINAL_MODEL_PATH,
            export=True, 

        )
        print("Model export to ONNX initiated by Optimum.")

        print(f"Saving ONNX model and tokenizer to: {ONNX_MODEL_SAVE_PATH}")
        ort_model.save_pretrained(ONNX_MODEL_SAVE_PATH)
        tokenizer.save_pretrained(ONNX_MODEL_SAVE_PATH)

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