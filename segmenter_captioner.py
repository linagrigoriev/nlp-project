import torch
import cv2
import numpy as np
import os
from PIL import Image
import easyocr
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration  

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("D:/nlp_project/nlp-collateral-description-agent/models/yolov8s-seg.pt")   #change this
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


ocr_reader = easyocr.Reader(['en', 'hu'], gpu=True)  

def extract_segments(image: Image.Image):
    """ Extracts segmented objects using YOLOv8 segmentation. """
    results = yolo_model(image, task="segment", conf=0.4)
    
    segments = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for idx, box in enumerate(results[0].boxes):
            obj_name = results[0].names[int(box.cls)]
            
            # Extract the mask and apply it
            mask = masks[idx]
            segmented_region = apply_mask(image, mask)
            
            segments.append({"object": obj_name, "cropped": segmented_region})
    
    return segments

def apply_mask(image: Image.Image, mask: np.ndarray):
    """ Applies a segmentation mask to extract the object. """
    np_image = np.array(image)
    np_image[mask == 0] = 0  # Set background pixels to black
    
    return Image.fromarray(np_image)

def generate_caption(image: Image.Image) -> str:
    """ Generates a caption for a given segmented object using BLIP. """
    inputs = blip_processor(image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs, max_length=50)
    return blip_processor.decode(outputs[0], skip_special_tokens=True)

def extract_text(image_path):
    """ Extracts text from the image using EasyOCR. """
    img = cv2.imread(image_path)
    results = ocr_reader.readtext(img)
    detected_texts = [res[1] for res in results]  
    
    return ", ".join(detected_texts) if detected_texts else "No text detected"

# **Process One Image**
image_path = "D:/nlp_project/nlp-collateral-description-agent/images/158220/IMG_20240319_141855_anonimized.jpg"  #change this
image = Image.open(image_path).convert("RGB").resize((640, 480))

# Extract text from the original image
detected_text = extract_text(image_path)

# Extract segmented objects
segments = extract_segments(image)

for segment in segments:
    caption = generate_caption(segment["cropped"])  
    
    # Merge detected text into caption
    enriched_caption = f"{caption} Detected object: {segment['object']}. Text found: {detected_text}."
    
    print(enriched_caption)
    segment["cropped"].show(title=segment["object"])
