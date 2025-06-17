import openai
import base64
import os
import re
import csv
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY_R")
BASE_URL = os.getenv("OPENAI_BASE_URL_R")
MODEL = os.getenv("OPENAI_MODEL_R")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Paths
IMAGES_DIR = "images"
MARKDOWN_FILE = "images/annotations.md" 
OUTPUT_DIR = "output/captions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper: Convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Generate caption
def generate_caption_and_text(image_path):
    base64_image = image_to_base64(image_path)

    prompt = (
        "You are an expert automotive appraiser. Carefully analyze the image and return plain text with two fields:\n\n"
        "Caption:\n"
        "- Concise, factual description of visible vehicle part(s) and their condition.\n"
        "- Use terms like: intact, damaged, worn, dirty, rusted.\n"
        "- Describe the number of parts seen.\n"
        "- Examples: 'climate control and media interface present in good condition', "
        "'dashboard panel damaged with cracks'\n\n"
        "Extracted Text:\n"
        "- All legible text: labels, stickers, serial numbers, UI elements, documents, etc.\n"
        "- Comma-separated or line-by-line.\n"
        "- Text can be in English, Hungarian, Chinese or more.\" \n\n"
        "Return only plain text. No markdown formatting. Only these two sections:\n"
        "Caption:\n...\nExtracted Text:\n..."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    # Sanitize markdown artifacts (e.g., '**Caption**:')
    content = re.sub(r"\*\*", "", content)

    # Parse Caption
    caption_match = re.search(r"Caption:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
    caption = caption_match.group(1).strip() if caption_match else ""

    # Parse Extracted Text
    text_match = re.search(r"Extracted Text:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
    extracted_text = text_match.group(1).strip() if text_match else ""

    # Fallbacks in case of model errors
    if not caption:
        caption = "[caption missing]"
    if not extracted_text:
        extracted_text = "[no text extracted]"

    return caption, extracted_text

# Helper: Parse training folder names from markdown headers
def get_train_folders_from_md(md_file):
    train_folders = set()
    with open(md_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^##\s+(.+)", line)
            if match:
                folder_name = match.group(1).strip()
                train_folders.add(folder_name)
    return train_folders

# Process images in each training folder
def process_folders(train_folders):
    all_folders = [f for f in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, f))]
    test_folders = [f for f in all_folders if f not in train_folders]

    print(f"\nTraining folders: {sorted(train_folders)}")
    print(f"Testing folders: {sorted(test_folders)}\n")

    for folder in train_folders:
        folder_path = os.path.join(IMAGES_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"Folder {folder} not found, skipping.")
            continue

        output_csv = os.path.join(OUTPUT_DIR, f"{folder}.csv")

        with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["image_filename", "caption", "extracted_text"])

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(folder_path, filename)
                    print(f"Processing {filename} in {folder}...")

                    try:
                        caption, extracted_text = generate_caption_and_text(image_path)
                        writer.writerow([filename, caption, extracted_text])
                        print(f"    Caption for {filename}: {caption}")
                    except Exception as e:
                        print(f"    Error processing {filename}: {e}")

# === Main Execution ===
if __name__ == "__main__":
    train_folders = get_train_folders_from_md(MARKDOWN_FILE)
    process_folders(train_folders)
