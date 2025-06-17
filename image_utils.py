import numpy as np
from PIL import Image
from model_loader import yolo_model
from model_loader import sam_predictor
import cv2

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

# def generate_caption(image: Image.Image):
#     inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
#     outputs = blip_model.generate(**inputs, max_length=50)
#     caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
#     return caption
    
# Gulji's functions
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