# src/processing/quantize_onnx_model.py
import os
import time
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

ORIGINAL_MODEL_PATH = "/mnt/object-store-persist-group36/model/Legal-BERT-finetuned"


LOCAL_OPTIMIZED_MODELS_DIR = "/tmp/optimized_models" # Parent directory for optimized models
ONNX_UNQUANTIZED_SUBDIR = "legal_bert_finetuned_onnx_unquantized" # For the intermediate ONNX model
ONNX_UNQUANTIZED_SAVE_PATH = os.path.join(LOCAL_OPTIMIZED_MODELS_DIR, ONNX_UNQUANTIZED_SUBDIR)


ONNX_INT8_QUANTIZED_SUBDIR = "legal_bert_finetuned_onnx_int8_quantized"
ONNX_INT8_QUANTIZED_SAVE_PATH = os.path.join(LOCAL_OPTIMIZED_MODELS_DIR, ONNX_INT8_QUANTIZED_SUBDIR)


os.makedirs(ONNX_UNQUANTIZED_SAVE_PATH, exist_ok=True)
os.makedirs(ONNX_INT8_QUANTIZED_SAVE_PATH, exist_ok=True)

print(f"Original PyTorch Model Path: {ORIGINAL_MODEL_PATH}")
print(f"Intermediate Unquantized ONNX Model Save Path: {ONNX_UNQUANTIZED_SAVE_PATH}")
print(f"Final INT8 Quantized ONNX Model Save Path: {ONNX_INT8_QUANTIZED_SAVE_PATH}")

def main():

    print("\n--- Starting Step 1: Exporting original model to standard ONNX format ---")
    export_start_time = time.time()
    try:
        print(f"Loading tokenizer from original model: {ORIGINAL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH)
        
        print(f"Exporting model from '{ORIGINAL_MODEL_PATH}' to ONNX format at '{ONNX_UNQUANTIZED_SAVE_PATH}'...")

        ort_model_unquantized = ORTModelForFeatureExtraction.from_pretrained(
            ORIGINAL_MODEL_PATH,
            export=True,
        )
        print("Saving unquantized ONNX model and tokenizer...")
        ort_model_unquantized.save_pretrained(ONNX_UNQUANTIZED_SAVE_PATH)
        tokenizer.save_pretrained(ONNX_UNQUANTIZED_SAVE_PATH) # Save tokenizer with it
        export_end_time = time.time()
        print(f"Unquantized ONNX model saved successfully to {ONNX_UNQUANTIZED_SAVE_PATH}")
        print(f"ONNX export took {export_end_time - export_start_time:.2f} seconds.")
    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")
        return # Stop if export fails


    print("\n--- Starting Step 2: Applying Dynamic INT8 Quantization to ONNX model ---")
    quantize_start_time = time.time()
    try:
        
        
        unquantized_onnx_model_path_for_quantizer = os.path.join(ONNX_UNQUANTIZED_SAVE_PATH, "model.onnx")
        quantized_onnx_model_output_path = os.path.join(ONNX_INT8_QUANTIZED_SAVE_PATH, "model_quantized.onnx")


        print(f"Initializing ORTQuantizer for dynamic quantization...")
 
        
        dynamic_quantizer = ORTQuantizer.from_pretrained(ONNX_UNQUANTIZED_SAVE_PATH)

        print("Creating dynamic quantization configuration (operators: MatMul, Add)...")
     
        
        dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False, operators_to_quantize=["MatMul", "Add"])


        print("Applying dynamic quantization...")
        dynamic_quantizer.quantize(
            save_dir=ONNX_INT8_QUANTIZED_SAVE_PATH, 
            quantization_config=dqconfig,

        )
        

        
        print(f"Copying tokenizer and config files to {ONNX_INT8_QUANTIZED_SAVE_PATH}...")
        for filename in ["config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]:
            src_file = os.path.join(ONNX_UNQUANTIZED_SAVE_PATH, filename)
            dst_file = os.path.join(ONNX_INT8_QUANTIZED_SAVE_PATH, filename)
            if os.path.exists(src_file):
                import shutil
                shutil.copy2(src_file, dst_file)
        
        quantize_end_time = time.time()
        print(f"Dynamic INT8 quantized ONNX model saved successfully to {ONNX_INT8_QUANTIZED_SAVE_PATH}")
        print(f"Quantization took {quantize_end_time - quantize_start_time:.2f} seconds.")

        print("\nContents of the INT8 Quantized ONNX model directory:")
        for item in os.listdir(ONNX_INT8_QUANTIZED_SAVE_PATH):
            print(f"- {item}") # Should include model.onnx (quantized)

    except Exception as e:
        print(f"An error occurred during ONNX quantization: {e}")
        print("Troubleshooting tips:")
        print("- Ensure the unquantized ONNX model was created successfully in the previous step.")
        print("- Check if ONNX_UNQUANTIZED_SAVE_PATH points to the correct directory.")

if __name__ == "__main__":
    main()