import pdfplumber
import os
import pandas as pd
import re

RAW_DIR = "/data/raw_pdfs/"
OUTPUT_DIR = "/data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "processed_cases.csv")

cases = []

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".pdf"):
        path = os.path.join(RAW_DIR, filename)
        
        with pdfplumber.open(path) as pdf:
            full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        # Extract metadata (basic)
        case_name = re.search(r"([A-Z][\w\s.,\-]+ v\. [A-Z][\w\s.,\-]+)", full_text)
        court = re.search(r"(Supreme Court|Court of Appeals|District Court|Family Court|Circuit Court|Appellate Division)", full_text)
        date = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", full_text)

        case_name = case_name.group(0) if case_name else "Unknown Case"
        court = court.group(0) if court else "Unknown Court"
        date = date.group(0) if date else "Unknown Date"

        # Add to the cases list
        cases.append([case_name, court, date, full_text[:1000]])

# Save to CSV
df = pd.DataFrame(cases, columns=["case_name", "court", "date", "summary"])
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Processed {len(df)} cases into {OUTPUT_FILE}")
