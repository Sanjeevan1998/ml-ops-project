import os
import re
import csv
import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
PDF_DIR = "./data"
OUTPUT_DIR = "./data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = "./case_metadata.csv"
MODEL_NAME = "openlm-research/open_llama_3b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Llama Model and Tokenizer
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)


def extract_text_from_pdf(pdf_file):
    """Extracts text from a single PDF file."""
    with fitz.open(pdf_file) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def generate_metadata(text):
    """Generates structured metadata from raw case text using Llama."""
    prompt = f"Extract the following metadata from this legal case document:\n1. Case Name\n2. Court\n3. Date of Decision\n4. Docket Number or Citation\n5. Judges Involved\n6. Summary of the Case\n\nDocument Text:\n{text}\n"  
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    outputs = model.generate(**inputs, max_length=4096, temperature=0.2)
    metadata_raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Extract structured metadata
    metadata_lines = metadata_raw.split("\n")
    case_name = court = date = docket_number = judges = summary = ""
    for line in metadata_lines:
        if line.startswith("Case Name:"):
            case_name = line.replace("Case Name:", "").strip()
        elif line.startswith("Court:"):
            court = line.replace("Court:", "").strip()
        elif line.startswith("Date of Decision:"):
            date = line.replace("Date of Decision:", "").strip()
        elif line.startswith("Docket Number or Citation:"):
            docket_number = line.replace("Docket Number or Citation:", "").strip()
        elif line.startswith("Judges Involved:"):
            judges = line.replace("Judges Involved:", "").strip()
        elif line.startswith("Summary of the Case:"):
            summary = line.replace("Summary of the Case:", "").strip()
    
    return [case_name, court, date, docket_number, judges, summary]


def process_all_pdfs(pdf_dir):
    """Processes all PDF files in the specified directory."""
    metadata_list = []
    for pdf_file in tqdm(os.listdir(pdf_dir)):
        if pdf_file.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_dir, pdf_file)
            raw_text = extract_text_from_pdf(full_path)
            metadata = generate_metadata(raw_text)
            if metadata:
                metadata_list.append([pdf_file] + metadata)
    return metadata_list


def save_metadata(metadata_list, output_file):
    """Saves the extracted metadata to a CSV file."""
    headers = ["File Name", "Case Name", "Court", "Date of Decision", "Docket Number", "Judges Involved", "Summary"]
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(metadata_list)


def main():
    print("Processing PDF files...")
    metadata_list = process_all_pdfs(PDF_DIR)
    save_metadata(metadata_list, OUTPUT_FILE)
    print(f"Metadata saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

