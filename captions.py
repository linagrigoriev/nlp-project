import torch
import cv2
import numpy as np
import os
import csv
from PIL import Image
import easyocr
import spacy
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
yolo_model = YOLO("yolov8s-seg.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
ocr_reader = easyocr.Reader(['en', 'hu'], gpu=torch.cuda.is_available())
nlp = spacy.load("en_core_web_sm")

# Utility Functions
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

def semantic_consistency_check(caption: str, detected_class: str) -> str:
    return "yes" if detected_class.lower() in caption.lower() else "no"

def multilabel_mislabel_check(caption: str, class_name: str) -> str:
    doc = nlp(caption.lower())
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return "mismatch" if class_name.lower() not in nouns else "ok"

def get_training_ids(annotations_path: str):
    """
    Reads the annotations.md file and returns a list of object IDs
    that are specified as level-2 headings (e.g., lines starting with "## ").
    """
    training_ids = []
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Check for level-2 markdown headings
                if line.startswith("## "):
                    training_id = line[3:].strip()
                    if training_id:
                        training_ids.append(training_id)
    except Exception as e:
        print(f"Error reading annotations file: {e}")
    return training_ids

# Main function – processes subfolders whose names match the IDs in annotations.md
def caption_images_from_folder(root_folder: str, output_csv: str = "captions.csv"):
    # Read annotations.md from the root folder to determine which folders to process.
    annotations_path = os.path.join(root_folder, "annotations.md")
    training_ids = get_training_ids(annotations_path) if os.path.exists(annotations_path) else []
    if training_ids:
        print(f"Processing folders with these IDs from annotations.md: {training_ids}")
    else:
        print("No training IDs extracted from annotations.md. Nothing to process.")
        return

    # Get all subfolders in the images folder and filter to include only those in training_ids.
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    subfolders_to_process = [folder for folder in subfolders if os.path.basename(folder) in training_ids]

    if not subfolders_to_process:
        print("No valid subfolders found to process.")
        return

    rows = [["image_path", "object_class", "caption", "text_detected",
             "semantic_consistent", "multilabel_check"]]

    for subfolder in subfolders_to_process:
        folder_name = os.path.basename(subfolder)
        print(f"\nProcessing folder: {folder_name}")
        image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No images found in {subfolder}.")
            continue

        for idx, filename in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing image: {filename}")
            image_path = os.path.join(subfolder, filename)
            try:
                image = Image.open(image_path).convert("RGB").resize((640, 480))
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            image_caption = generate_caption(image)
            detected_text = extract_text(image_path)
            segments = extract_segments(image)

            rows.append([
                image_path, "full_image", image_caption, detected_text,
                "N/A", "N/A"
            ])

            for seg_idx, segment in enumerate(segments, 1):
                caption = generate_caption(segment["cropped"])
                obj_class = segment["object"]
                semantic_ok = semantic_consistency_check(caption, obj_class)
                multilabel_check = multilabel_mislabel_check(caption, obj_class)

                rows.append([
                    image_path, obj_class, caption, detected_text,
                    semantic_ok, multilabel_check
                ])

    with open(output_csv, mode="w", newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)

    print(f"\nFinished! Results saved to {output_csv}")

if __name__ == "__main__":
    caption_images_from_folder("images")
