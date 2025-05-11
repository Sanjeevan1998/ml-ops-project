
import os
import re
import csv
import fitz  # PyMuPDF
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Configuration
PDF_DIR = "/data"
OUTPUT_DIR = "/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

openai.api_key = OPENAI_API_KEY

def extract_text_from_pdf(pdf_file):
    """Extracts text from a single PDF file."""
    with fitz.open(pdf_file) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def generate_metadata(text):
    """Generates structured metadata from raw case text using OpenAI GPT-4."""
    prompt = f"""Extract the following metadata from this legal case document:
    1. Case Name
    2. Court
    3. Date of Decision
    4. Docket Number or Citation
    5. Judges Involved
    6. Summary of the Case

    Document Text:
    {text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2
        )
        metadata = response['choices'][0]['message']['content'].strip()
        return metadata
    except Exception as e:
        print(f"Error generating metadata: {e}")
        return None


def process_all_pdfs(pdf_dir):
    """Processes all PDF files in the specified directory."""
    metadata_list = []
    for pdf_file in tqdm(os.listdir(pdf_dir)):
        if pdf_file.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_dir, pdf_file)
            raw_text = extract_text_from_pdf(full_path)
            metadata = generate_metadata(raw_text)
            if metadata:
                # Extract individual fields from metadata
                metadata_list.append([pdf_file, metadata])
    return metadata_list


def save_metadata(metadata_list, output_file):
    """Saves the extracted metadata to a CSV file."""
    headers = ["File Name", "Metadata"]
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
