import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import re

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY_R")
BASE_URL = os.getenv("OPENAI_BASE_URL_R")
MODEL = os.getenv("OPENAI_MODEL_R")

# Initialize client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Output directory
output_dir = Path("output/generated_reports_raw")
output_dir.mkdir(exist_ok=True)

# Directory where script is located
script_dir = Path("output/captions")

# Find all evaluation CSV files matching the pattern
csv_files = list(script_dir.glob("*.csv"))

# Helper: construct prompt for a folder
def build_prompt(folder_name, captions):
    prompt = f"""Given the following image captions for a vehicle stored in folder {folder_name}, generate a structured markdown report with these required sections:

Important instructions:
- Use the exact model and manufacturer when available, without adding parentheses or explanatory notes.
- Write only the predominant manufacturer and specific model (if multiple are present).
- If a caption says "DAF XF 480", you must write **Model:** XF 480 (not just XF).
- If unsure in a **Model:** field, write Unspecified
- Write in a formal tone used in professional vehicle reports.
- Skip any non-confirmed models or brands that appear only once or ambiguously.

Start your output with these sections:

## {folder_name}

### Identification & General Data
- Asset Type
- Manufacturer
- Model
- Vehicle Type
- Year of Manufacture
- First Registration Date
- First Registration Country
- Engine Power
- Engine Displacement
- Environmental Classification
- Seating Capacity
- Transmission Type
- Technical Inspection Valid Until
- Odometer Reading
- Accepted Mileage
- Number of Keys
- Service Book
- Usage
- Document Date

### Inspection Methods
### Condition Assessment
### Valuation Principles
### Documentation & Accessories

Captions (from images):
{chr(10).join(f"- {cap}" for cap in captions)}

Write the markdown in the above format. In asset type write what type of vehicle it is. Use domain knowledge to fill in missing standard details when captions are incomplete, but mentioned that they are assumed or inferred.
"""
    return prompt.strip()

# Process each CSV
for csv_file in csv_files:
    folder_name = csv_file.stem

    df = pd.read_csv(csv_file)
    captions = []

    for _, row in df.iterrows():
        # if row.get('object_class') != 'full_image':
        #     continue  # Skip non-full_image entries

        caption = str(row.get('caption', '')).strip()
        text = str(row.get('extracted_text', '')).strip()

        # Skip rows with no useful data
        if not caption and not text:
            continue

        # Don't include 'No legible text is visible.' as actual text
        if text.lower() in {"No legible text is visible.", "none", "n/a"}:
            text = ""

        combined = f"{caption} [Detected Text: {text}]" if text else caption
        if combined.strip():
            captions.append(combined)

    if not captions:
        print(f"No captions found in {csv_file.name}. Skipping.")
        continue

    prompt = build_prompt(folder_name, captions)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional vehicle evaluator generating high-quality, detailed markdown reports for used commercial vehicles. Use the standard structure and style found in formal technical assessments, with thorough explanations, domain language, and explicit mention of missing or inferred data. Use markdown formatting and avoid overly generic phrasing."},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content

    output_path = output_dir / f"{folder_name}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Generated: {output_path.name}")
