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
output_dir = Path("generated_reports_full_image")
output_dir.mkdir(exist_ok=True)

# Directory where script is located
script_dir = Path(__file__).resolve().parent

# Find all evaluation CSV files matching the pattern
csv_files = list(script_dir.glob("evaluation_*.csv"))

# Helper: construct prompt for a folder
def build_prompt(folder_name, captions):
    prompt = f"""Given the following image captions for a vehicle stored in folder {folder_name}, generate a structured markdown report with these required sections:

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

Write the markdown in the above format. Use domain knowledge to fill in missing standard details when captions are incomplete, and indicate where information is inferred or missing explicitly.
"""
    return prompt.strip()

# Process each CSV
for csv_file in csv_files:
    folder_name_match = re.match(r"evaluation_(.+)\.csv", csv_file.name)
    if not folder_name_match:
        continue  # skip if filename doesn't match

    folder_name = folder_name_match.group(1)

    df = pd.read_csv(csv_file)
    captions = []

    for _, row in df.iterrows():
        if row.get('object_class') != 'full_image':
            continue  # Skip non-full_image entries

        caption = row.get('caption', '')
        text = row.get('text_detected', '')
        combined = f"{caption} [Detected Text: {text}]" if pd.notna(text) and str(text).strip() else caption
        if combined.strip():
            captions.append(combined)

    if not captions:
        print(f"⚠️ No captions found in {csv_file.name}. Skipping.")
        continue

    prompt = build_prompt(folder_name, captions)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional vehicle evaluator. Return structured markdown reports."},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content

    output_path = output_dir / f"{folder_name}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Generated: {output_path.name}")
