
import os
import re
import csv
import fitz  # PyMuPDF
import spacy
from tqdm import tqdm

# Configuration
PDF_DIR = "./data"
OUTPUT_DIR = "./data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = "./case_metadata.csv"

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

def extract_text_from_pdf(pdf_file):
    """Extracts text from a single PDF file."""
    with fitz.open(pdf_file) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def extract_metadata(text):
    """Extracts metadata using spaCy NLP model."""
    doc = nlp(text)
    case_name = ""
    court = ""
    date = ""
    docket_number = ""
    summary = ""

    # Extract common metadata fields
    for ent in doc.ents:
        if ent.label_ == "ORG":
            court = ent.text
        elif ent.label_ == "DATE":
            date = ent.text
        elif ent.label_ == "GPE":
            case_name = ent.text
        elif ent.label_ == "LAW":
            docket_number = ent.text

    # Basic summary extraction (first few lines)
    summary = " ".join(text.split("\n")[:5])
    
    return [case_name, court, date, docket_number, summary]


def process_all_pdfs(pdf_dir):
    """Processes all PDF files in the specified directory."""
    metadata_list = []
    for pdf_file in tqdm(os.listdir(pdf_dir)):
        if pdf_file.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_dir, pdf_file)
            raw_text = extract_text_from_pdf(full_path)
            metadata = extract_metadata(raw_text)
            if metadata:
                metadata_list.append([pdf_file] + metadata)
    return metadata_list


def save_metadata(metadata_list, output_file):
    """Saves the extracted metadata to a CSV file."""
    headers = ["File Name", "Case Name", "Court", "Date", "Docket Number", "Summary"]
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
