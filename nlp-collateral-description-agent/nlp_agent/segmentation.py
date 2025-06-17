import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return 0 if union == 0 else intersection / union


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


class Segmenter:
    def __init__(self, yolo_path, sam_checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(yolo_path)
        sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint).to(self.device)
        self.sam_predictor = SamPredictor(sam)

    def extract_combined_mask(self, image: Image.Image):
        results = self.yolo_model(image, task="segment", conf=0.25)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

            # Convert masks to boolean masks with threshold 0.5
            bool_masks = [m > 0.5 for m in masks]

            # Group and merge masks with IoU > 0.7 for YOLO
            merged_masks = group_and_merge_masks(bool_masks, iou_threshold=0.7)

            # Filter out small masks
            filtered_masks = filter_small_masks(merged_masks)

            if not filtered_masks:
                return None

            # Merge all filtered masks into one combined mask (logical OR)
            combined_mask = np.zeros_like(filtered_masks[0], dtype=bool)
            for m in filtered_masks:
                combined_mask = np.logical_or(combined_mask, m)

            return apply_mask(image, combined_mask.astype(np.uint8))
        return None

    def run_sam_full_image(self, image: Image.Image):
        np_image = np.array(image)
        self.sam_predictor.set_image(np_image)

        height, width, _ = np_image.shape
        input_box = np.array([0, 0, width, height])

        masks, scores, _ = self.sam_predictor.predict(box=input_box, multimask_output=True)

        conf_threshold = 0.85
        valid_masks = masks[[i for i, score in enumerate(scores) if score > conf_threshold]]

        if len(valid_masks) == 0:
            return None

        merged_masks = group_and_merge_masks(valid_masks, iou_threshold=0.5)
        filtered_masks = filter_small_masks(merged_masks, min_area=5000)
        if not filtered_masks:
            return None

        largest_mask = max(filtered_masks, key=lambda m: m.sum()).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        blurred = cv2.GaussianBlur(clean_mask, (9, 9), 0)
        _, final_mask = cv2.threshold(blurred, 0.5, 1, cv2.THRESH_BINARY)

        segmented_img = np_image.copy()
        segmented_img[final_mask.astype(bool) == 0] = 0

        return Image.fromarray(segmented_img)
