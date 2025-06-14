import torch
import cv2
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything import SamPredictor, sam_model_registry
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load models
yolo_model = YOLO("D:/nlp_project/nlp-collateral-description-agent/models/yolov8x-seg.pt")

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


state_dict = torch.load("D:/nlp_project/nlp-collateral-description-agent/models/blip_finetuned.pt", map_location="cpu")
blip_model.load_state_dict(state_dict)

blip_model = blip_model.to(device)


blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
ocr_reader = easyocr.Reader(['en', 'hu'], gpu=torch.cuda.is_available())

sam_checkpoint = "D:/nlp_project/nlp-collateral-description-agent/models/sam_vit_l_0b3195.pth"
sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint).to(device)
sam_predictor = SamPredictor(sam)


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def group_and_merge_masks(masks, iou_threshold=0.5):
    groups = []
    used = set()

    for i in range(len(masks)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(masks)):
            if j in used:
                continue
            if mask_iou(masks[i], masks[j]) > iou_threshold:
                group.append(j)
                used.add(j)
        groups.append(group)

    merged_masks = []
    for group in groups:
        merged_mask = np.zeros_like(masks[0], dtype=bool)
        for idx in group:
            merged_mask = np.logical_or(merged_mask, masks[idx])
        merged_masks.append(merged_mask)

    return merged_masks


def filter_small_masks(masks, min_area=2000):
    return [m for m in masks if m.sum() >= min_area]


def apply_mask(image: Image.Image, mask: np.ndarray):
    np_image = np.array(image)
    np_image[mask == 0] = 0
    return Image.fromarray(np_image)


def extract_combined_mask(image: Image.Image):
    results = yolo_model(image, task="segment", conf=0.25)
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        combined_mask = np.any(masks > 0.5, axis=0)
        combined_mask_uint8 = combined_mask.astype(np.uint8)
        return apply_mask(image, combined_mask_uint8)
    return None


def run_sam_full_image(image: Image.Image):
    np_image = np.array(image)
    sam_predictor.set_image(np_image)

    height, width, _ = np_image.shape
    input_box = np.array([0, 0, width, height])  

    masks, scores, _ = sam_predictor.predict(
        box=input_box,
        multimask_output=True
    )

    # Filter masks by confidence threshold
    conf_threshold = 0.85
    valid_indices = [i for i, score in enumerate(scores) if score > conf_threshold]
    valid_masks = masks[valid_indices]

    if len(valid_masks) == 0:
        print("No SAM masks passed confidence threshold!")
        return None

    
    merged_masks = group_and_merge_masks(valid_masks, iou_threshold=0.5)

    
    filtered_masks = filter_small_masks(merged_masks, min_area=5000)
    if not filtered_masks:
        print("No SAM masks passed area threshold!")
        return None

    # Pick largest mask
    largest_mask = max(filtered_masks, key=lambda m: m.sum())

    # Morphological clean-up on largest mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    largest_mask_uint8 = largest_mask.astype(np.uint8)

    # Remove small holes and noise (opening + closing)
    clean_mask = cv2.morphologyEx(largest_mask_uint8, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Gaussian blur + re-threshold to smooth edges
    blurred = cv2.GaussianBlur(clean_mask, (9, 9), 0)
    _, final_mask = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)
    final_mask = final_mask.astype(bool)

    # Apply mask to image
    segmented_img = np_image.copy()
    segmented_img[~final_mask] = 0

    return Image.fromarray(segmented_img)


def generate_caption(image: Image.Image) -> str:
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(**inputs, max_length=50)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption


def extract_text(image_path):
    img = cv2.imread(image_path)
    results = ocr_reader.readtext(img)
    detected_texts = [res[1] for res in results]
    return ", ".join(detected_texts) if detected_texts else " "


def process_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((640, 480))
    
    detected_text = extract_text(image_path)
    combined_segment = extract_combined_mask(image)

    if combined_segment:
        caption = generate_caption(combined_segment)
    else:
        sam_segment = run_sam_full_image(image)
        if sam_segment is not None:
            caption = generate_caption(sam_segment)
        else:
            caption = "Segmentation failed."
    
    enriched_caption = f"{caption}. Text found: {detected_text}."
    return enriched_caption


if __name__ == "__main__":
    folder_path = "your_dir/images"   #change the path
    output_file = "your_dir/captions_report.txt"    #change the path

    # Supported image extensions
    supported_exts = {".jpg", ".jpeg", ".png"}

    with open(output_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_exts:
                continue

            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")
            enriched_caption = process_image(image_path)
            
            f.write(f"{enriched_caption}\n")

    print(f"Processing complete. Captions saved to {output_file}.")
