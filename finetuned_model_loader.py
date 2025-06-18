import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import easyocr
import spacy
from config import DEVICE, FINETUNED_PATH
import os
from segment_anything import SamPredictor, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
yolo_model = YOLO("yolov8s-seg.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

sam_checkpoint = "sam_vit_l_0b3195.pth"
sam = sam_model_registry["vit_l"](checkpoint=sam_checkpoint).to(device)
sam_predictor = SamPredictor(sam)

# Optional fine-tuned weights
if os.path.exists(FINETUNED_PATH):
    blip_model.load_state_dict(torch.load(FINETUNED_PATH, map_location=DEVICE))

ocr_reader = easyocr.Reader(['en', 'hu'], gpu=torch.cuda.is_available())
nlp = spacy.load("en_core_web_sm")
