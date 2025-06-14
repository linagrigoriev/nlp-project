import os
import csv
import difflib
import torch
import cv2
import numpy as np
from PIL import Image
import easyocr
import spacy
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, get_linear_schedule_with_warmup, pipeline

# For API calls and environment variables
import pandas as pd
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import re
import time
from difflib import SequenceMatcher
import json

SIMILARITY_LOSS_WEIGHT = 2.0  

# ------------------ Model & Utility Loading ------------------ #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_similarity_penalty(new_caption: str, previous_captions: list, threshold: float = 0.6) -> float:
    for prev in previous_captions:
        ratio = SequenceMatcher(None, new_caption.lower(), prev.lower()).ratio()
        # print(f"    ↪ Similarity to \"{prev[:60]}...\": {ratio:.4f}")  # Debug line
        if ratio > threshold:
            penalty = ratio
            # print(f"    ↪ Applying penalty: {penalty:.4f} (ratio: {ratio:.4f})")
            return penalty
    return 0.0

# Vision/AI Models
yolo_model = YOLO("yolov8s-seg.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Fine-tuned weights path
finetuned_path = os.path.join("images", "blip_finetuned.pt")

# Ground truth captions path
captions_path = os.path.join("images", "ground_truth_captions.json")
captions_path_val = os.path.join("images", "ground_truth_captions_val.json")

# Summarizer
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Conditional load
if os.path.exists(finetuned_path):
    print(f"Loading fine-tuned BLIP model from {finetuned_path}")
    blip_model.load_state_dict(torch.load(finetuned_path, map_location=device))
else:
    print("Fine-tuned BLIP model not found. Using base model.")

ocr_reader = easyocr.Reader(['en', 'hu'], gpu=torch.cuda.is_available())
nlp = spacy.load("en_core_web_sm")

# ------------------ Image Processing Functions ------------------ #
def apply_mask(image: Image.Image, mask: np.ndarray):
    np_image = np.array(image)
    np_image[mask == 0] = 0
    return Image.fromarray(np_image)

def extract_segments(image: Image.Image):
    results = yolo_model(image, task="segment", conf=0.5)
    segments = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for idx, box in enumerate(results[0].boxes):
            obj_name = results[0].names[int(box.cls)]
            mask = masks[idx]
            segmented_region = apply_mask(image, mask)
            segments.append({"object": obj_name, "cropped": segmented_region})
    return segments

def generate_caption(image: Image.Image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, max_length=50)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def extract_text(image_path):
    img = cv2.imread(image_path)
    results = ocr_reader.readtext(img)
    detected_texts = [res[1] for res in results]
    return ", ".join(detected_texts) if detected_texts else "No text detected"

# ------------------ Annotations Parsing Functions ------------------ #
def get_training_ids(annotations_path: str):
    """
    Reads annotations.md and returns a list of folder names extracted from level‑2 headings.
    """
    training_ids = []
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("## "):
                    folder_id = line[3:].strip()
                    if folder_id:
                        training_ids.append(folder_id)
    except Exception as e:
        print(f"Error reading annotations file: {e}")
    return training_ids

def parse_annotations(annotations_path: str):
    """
    Parses the annotations.md file into a dict mapping folder name to its full markdown content.
    """
    annotations = {}
    current_folder = None
    current_lines = []
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("## "):
                    if current_folder:
                        annotations[current_folder] = "\n".join(current_lines)
                    current_folder = line[3:].strip()
                    current_lines = [line.strip()]
                else:
                    current_lines.append(line.strip())
            if current_folder:
                annotations[current_folder] = "\n".join(current_lines)
    except Exception as e:
        print(f"Error parsing annotations: {e}")
    return annotations

def remove_determined_values(text: str) -> str:
    """
    Cleans markdown text by:
    - Removing the 'Determined Values' section.
    - Stripping markdown symbols (###, **, -, newlines).
    - Keeping only lines that describe the object and its condition.
    """
    lines = text.splitlines()
    cleaned_lines = []
    skip = False

    for line in lines:
        line = line.strip()

        # Skip the "Determined Values" section
        if line.startswith("### Determined Values"):
            skip = True
            continue
        if skip and line.startswith("### "):
            skip = False

        if skip or not line:
            continue

        # Remove markdown symbols and formatting
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)  # Remove bold
        line = re.sub(r"^- ", "", line)               # Remove list dashes
        line = re.sub(r"###", "", line)               # Remove headers
        line = line.strip()

        # Keep lines that describe the object or its state
        if any(keyword in line.lower() for keyword in [
            "vehicle", "truck", "car", "machine", "container", "unit", "equipment", 
            "damaged", "intact", "burnt", "destroyed", "missing", "functional", "partial"
        ]):
            cleaned_lines.append(line)

    return " ".join(cleaned_lines)

# ------------------ API Prompt & Comparison Functions ------------------ #
def build_prompt(folder_name, captions):
    nl = "\n"  # Define the newline string outside the f-string
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
{nl.join(f"- {cap}" for cap in captions)}

Use domain knowledge to fill in missing details where possible; indicate where information is inferred or missing.
"""
    return prompt.strip()

def generate_report(folder_name, captions):
    prompt = build_prompt(folder_name, captions)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional vehicle evaluator. Return structured markdown reports in the specified format."},
            {"role": "user", "content": prompt}
        ]
    )
    report = response.choices[0].message.content
    return report

def compare_reports(generated: str, original: str) -> str:
    gen_lines = generated.splitlines(keepends=True)
    orig_lines = original.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, gen_lines, fromfile='Original', tofile='Generated', lineterm="")
    return "\n".join(diff)

# ------------------ OpenAI API Client Setup ------------------ #
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY_R")
BASE_URL = os.getenv("OPENAI_BASE_URL_R")
MODEL = os.getenv("OPENAI_MODEL_R")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ------------------ Training Routine ------------------ #
def clean_duplicate_phrases(text: str) -> str:
    """
    Removes duplicated noun phrases like "a black wheel and a black wheel".
    """
    phrases = re.split(r",| and ", text)
    seen = set()
    cleaned_phrases = []
    for phrase in phrases:
        norm = phrase.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            cleaned_phrases.append(phrase.strip())
    return ", ".join(cleaned_phrases)

def remove_repeated_parts(text: str) -> str:
    """
    Removes repeated parts.
    """
    # Trim trailing commas or punctuation from cleanup
    return re.sub(r",+\s*", ", ", text).strip(",. ")

def refine_caption(image_caption: str, annotation_summary: str):
    image_caption = remove_repeated_parts(image_caption)
    cleaned_caption = clean_duplicate_phrases(image_caption)

    prompt = f"""
You are an expert reviewer of machine-generated image captions. Your task is to:

- Verify if any damage, defects, or specific states (e.g., flat tire, broken windshield) mentioned in the caption are clearly supported by the visible evidence or the annotation summary.
- If such claims are NOT supported, remove or correct them.
- Keep the caption concise and truthful based only on visible evidence or annotations.
- Avoid adding any new technical details, brand names, or assumptions.
- Neutral, descriptive details about visible components (e.g., GPS device, car window, dashboard) or tools are allowed and should be preserved.
- Do NOT mention any colors in your caption. Ignore color information entirely.
- Ignore black squares or masked areas added to hide sensitive data; do not mention them explicitly.
- If the caption mentions a piece of paper, document, plates or similar item, do NOT reinterpret it as anything else—accept it as is.
- Maintain natural, clear language.

Original Caption:
"{cleaned_caption}"

Annotation Summary:
"{annotation_summary}"

Based on the above, provide the corrected and verified caption only.
"""

    print("Sending request to OpenAI API...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You carefully verify and refine captions to ensure factual correctness and avoid hallucinated defects or brands."
            },
            {"role": "user", "content": prompt}
        ]
    )

    refined = response.choices[0].message.content.strip()
    refined = remove_repeated_parts(refined)
    return clean_duplicate_phrases(refined)

def training_loop(root_folder: str = "images", output_csv: str = "captions_blip.csv"):
    annotations_path = os.path.join(root_folder, "annotations.md")
    if not os.path.exists(annotations_path):
        print("annotations.md not found in the images folder.")
        return

    # Parse the original annotations and prepare targets
    original_annotations = parse_annotations(annotations_path)
    # Use the annotations without "Determined Values" as target captions.
    for key in original_annotations:
        original_annotations[key] = remove_determined_values(original_annotations[key])
        # print(f"Key: {key}, orifinal annotation: {original_annotations}")
    folder_ids = list(original_annotations.keys())
    if not folder_ids:
        print("No valid folder IDs found in annotations.md")
        return
    
    if os.path.exists(captions_path):
        with open(captions_path, "r") as f:
            refined_captions = json.load(f)
    else:
        refined_captions = {}
        print("Captions not found. They will be generated during the first epoch")
    
    if os.path.exists(captions_path_val):
        with open(captions_path_val, "r") as f:
            refined_captions_val = json.load(f)
    else:
        refined_captions_val = {}
        print("Captions for val set not found. They will be generated during the first epoch")

    # Determine available subfolders in the images folder that match the annotations.
    available_folders = []
    test_folders = [] 
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path) and folder in folder_ids:
            available_folders.append(folder_path)
        else:
            # Not in folder_ids — add to test_folders
            test_folders.append(folder_path)

    available_folders.sort()

    if len(available_folders) < 3:
        print("Need at least 3 folders (for training, validation, and test).")
        return

    # Reserve one folder for validation and one for test; all others for training.
    validation_folder = available_folders[0]
    training_folders = available_folders[1:]

    print("Folder split:")
    print("  Training folders:", [os.path.basename(f) for f in training_folders])
    print("  Validation folder:", os.path.basename(validation_folder))
    print("  Test folder:", [os.path.basename(f) for f in test_folders])

    # Setup optimizer and scheduler to fine-tune BLIP (we update only the captioning model)
    num_epochs = 2
    optimizer = torch.optim.Adam(blip_model.parameters(), lr=1e-6)
    # Count total number of training steps (rough estimate)
    total_steps = sum([len(os.listdir(f)) for f in training_folders if os.listdir(f)]) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    # For collecting outputs (for final CSV)
    all_rows = []

    # Begin training epochs
    for epoch in range(1, num_epochs + 1):

        blip_model.train()
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        total_train_loss = 0.0
        train_image_count = 0

        # Gather all training images from all folders
        training_samples = []
        for folder in training_folders:
            folder_name = os.path.basename(folder)
            target_caption = original_annotations.get(folder_name)
            if target_caption is None:
                continue
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder, filename)
                    training_samples.append((image_path, folder_name, target_caption))

        # Shuffle training samples
        np.random.shuffle(training_samples)
        # Train on each image independently
        # for image_index, (image_path, folder_name, target_caption) in enumerate(training_samples, start=1):
        #     print(f"\n  Training image {image_index}/{len(training_samples)} from folder: {folder_name} - {os.path.basename(image_path)}")
        #     try:
        #         image = Image.open(image_path).convert("RGB").resize((640, 480))
        #     except Exception as e:
        #         print(f"Failed to open {image_path}: {e}")
        #         continue

        #     # Step 1: Generate caption
        #     blip_model.eval()
        #     with torch.no_grad():
        #         inference_inputs = blip_processor(images=image, return_tensors="pt").to(device)
        #         generated_ids = blip_model.generate(**inference_inputs)
        #         predicted_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        #     blip_model.train()

        #     print(f"    Generated caption: {predicted_caption}")

        #     # Step 2: Refine with OpenAI
        #     if image_path in refined_captions:
        #         refined_caption = refined_captions[image_path]
        #         print(f"    Using cached refined caption: {refined_caption}")
        #     else:
        #         try:
        #             start_time = time.time()
        #             refined_caption = refine_caption(predicted_caption, target_caption)
        #             print(f"    Refined caption (in {time.time() - start_time:.2f}s): {refined_caption}")

        #         except Exception as e:
        #             print(f"Refinement error: {e}")
        #             refined_caption = predicted_caption

        #     # === Step 3: Use the refined caption for training ===
        #     inputs = blip_processor(
        #         images=image,
        #         text=refined_caption,
        #         return_tensors="pt",
        #         truncation=True,
        #         max_length=512
        #     )
        #     inputs = {k: v.to(device) for k, v in inputs.items()}
        #     labels = inputs["input_ids"].clone()
        #     labels[labels == blip_processor.tokenizer.pad_token_id] = -100

        #     outputs = blip_model(**inputs, labels=labels)
        #     loss = outputs.loss

        #     # Apply caption repetition penalty
        #     # similarity_penalty = caption_similarity_penalty(
        #     #     refined_caption,
        #     #     [cap for key, cap in refined_captions.items() if key.startswith(f"{folder_name}/")]
        #     # )
        #     # scaled_penalty = SIMILARITY_LOSS_WEIGHT * similarity_penalty
        #     total_loss = loss

        #     # print(f"    Cross-entropy loss: {loss.item():.4f}")
        #     # print(f"    Regularization penalty: {scaled_penalty:.4f} (raw: {similarity_penalty:.4f} × weight {SIMILARITY_LOSS_WEIGHT})")
        #     # print(f"    Total loss: {total_loss.item():.4f}")


        #     # Backprop and step
        #     total_loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()

        #     total_train_loss += loss.item()
        #     train_image_count += 1

        #     refined_captions[image_path] = refined_caption


        #     # === Step 4: Log all results ===
        #     all_rows.append([
        #         image_path,
        #         folder_name,
        #         target_caption,
        #         predicted_caption,
        #         refined_caption
        #     ])

        # avg_train_loss = total_train_loss / train_image_count if train_image_count else 0.0
        # print(f"Epoch {epoch} completed. Average training loss: {avg_train_loss:.4f}")

        # # Optionally, save the fine-tuned model checkpoint.
        # model_save_path = os.path.join(root_folder, "blip_finetuned.pt")
        # torch.save(blip_model.state_dict(), model_save_path)
        # print(f"Fine-tuned BLIP model saved to {model_save_path}.")

        # ---- Validation Step (evaluate on the reserved validation folder) ---- #
        blip_model.eval()
        val_loss = 0.0
        val_image_count = 0
        validation_captions = []

        val_folder_name = os.path.basename(validation_folder)
        target_caption_val = original_annotations.get(val_folder_name, "No Target")

        image_files = [f for f in os.listdir(validation_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_val_images = len(image_files)
        print(f"\nProcessing validation folder '{val_folder_name}' with {total_val_images} images.")

        for val_index, filename in enumerate(image_files, start=1):
            image_path = os.path.join(validation_folder, filename)
            print(f"  Validation: processing image {val_index}/{total_val_images}: {filename}")
            try:
                image = Image.open(image_path).convert("RGB").resize((640, 480))
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            # === Step 1: Generate caption with BLIP ===
            with torch.no_grad():
                inference_inputs = blip_processor(images=image, return_tensors="pt").to(device)
                generated_ids = blip_model.generate(**inference_inputs)
                predicted_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            # === Step 2: Refine caption using OpenAI + annotations ===
            print(f"    Generated caption: {predicted_caption}")

            # Step 2: Refine with OpenAI
            if image_path in refined_captions_val:
                refined_caption = refined_captions_val[image_path]
                print(f"    Using cached refined caption: {refined_caption}")
            else:
                try:
                    start_time = time.time()
                    refined_caption = refine_caption(predicted_caption, target_caption_val)
                    print(f"    Refined caption (in {time.time() - start_time:.2f}s): {refined_caption}")

                except Exception as e:
                    print(f"Refinement error: {e}")
                    refined_caption = predicted_caption

            # === Step 3: Compute validation loss using refined caption ===
            inputs = blip_processor(
                images=image,
                text=refined_caption,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                labels = inputs["input_ids"].clone()
                labels[labels == blip_processor.tokenizer.pad_token_id] = -100
                outputs = blip_model(**inputs, labels=labels)
                loss = outputs.loss

            val_loss += loss.item()
            val_image_count += 1

            # === Step 4: Log result ===
            validation_captions.append((image_path, predicted_caption, refined_caption))

            refined_captions_val[image_path] = refined_caption

        avg_val_loss = val_loss / val_image_count if val_image_count else 0.0
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Optionally print samples
        # for idx, (img_path, cap, ref_cap) in enumerate(validation_captions[:3], 1):
        #     print(f"Validation sample {idx}: {img_path}")
        #     print(f"Generated: {cap}")
        #     print(f"Refined:   {ref_cap}\n")

                # Save refined captions after epoch 1
        if epoch == 1:
            if not os.path.exists(captions_path):
                try:
                    with open(captions_path, 'w', encoding='utf-8') as f:
                        json.dump(refined_captions, f, indent=2, ensure_ascii=False)
                    print(f"Saved refined training captions to {captions_path}")
                except Exception as e:
                    print(f"Failed to save refined training captions: {e}")
            if not os.path.exists(captions_path_val):
                try:
                    with open(captions_path_val, 'w', encoding='utf-8') as f:
                        json.dump(refined_captions_val, f, indent=2, ensure_ascii=False)
                    print(f"Saved refined validation captions to {captions_path_val}")
                except Exception as e:
                    print(f"Failed to save refined validation captions: {e}")

    # ---- Final Test Evaluation on the Test Folders ---- #
    # Loop through all test folders
    for test_folder in test_folders:
        print(f"\n--- Final Evaluation on Test Folder: {os.path.basename(test_folder)} ---")
        test_folder_name = os.path.basename(test_folder)
        target_caption_test = original_annotations.get(test_folder_name, "No Target")

        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_test_images = len(image_files)
        print(f"\nProcessing test folder '{test_folder_name}' with {total_test_images} images.")

        rows = [["image_path", "object_class", "caption", "text_detected"]]

        for test_index, filename in enumerate(image_files, start=1):
            image_path = os.path.join(test_folder, filename)
            print(f"  Test: processing image {test_index}/{total_test_images}: {filename}")
            try:
                image = Image.open(image_path).convert("RGB").resize((640, 480))
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            with torch.no_grad():
                inference_inputs = blip_processor(images=image, return_tensors="pt").to(device)
                generated_ids = blip_model.generate(**inference_inputs)
                predicted_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            detected_text = extract_text(image_path)
            segments = extract_segments(image)

            rows.append([
                image_path, "full_image", predicted_caption, detected_text
            ])

            for seg_idx, segment in enumerate(segments, 1):
                caption = generate_caption(segment["cropped"])
                obj_class = segment["object"]

                rows.append([
                    image_path, obj_class, caption, detected_text
                ])

        print(f"Processed {len(image_files)} test images.")

        # ---- Save Final CSV Output for this folder ---- #
        output_csv = f"evaluation_{test_folder_name}.csv"
        with open(output_csv, mode="w", newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

        print(f"\n✅ Finished! Results saved to {output_csv}")

if __name__ == "__main__":
    training_loop("images", "captions.csv")