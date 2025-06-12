import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL")

# Initialize client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Read the captions CSV
df = pd.read_csv("captions.csv")

# Group captions by folder (e.g., images\157515_v2)
folder_groups = defaultdict(list)

for _, row in df.iterrows():
    folder = Path(row['image_path']).parts[1]  # e.g., 157515_v2
    caption = row['caption']
    folder_groups[folder].append(caption)

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
### Determined Values
### Documentation & Accessories

Captions (from images):
{chr(10).join(f"- {cap}" for cap in captions)}

Write the markdown in the above format. Use domain knowledge to fill in missing standard details when captions are incomplete, and indicate where information is inferred or missing explicitly.
"""
    return prompt.strip()

# Output directory
output_dir = Path("generated_reports")
output_dir.mkdir(exist_ok=True)

# Process each folder
for folder, captions in folder_groups.items():
    prompt = build_prompt(folder, captions)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional vehicle evaluator. Return structured markdown reports."},
            {"role": "user", "content": prompt}
        ]
    )

    report = response.choices[0].message.content

    with open(output_dir / f"{folder}.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Generated: {folder}.md")

