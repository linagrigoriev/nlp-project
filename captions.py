import os
from PIL import Image
import pandas as pd
import torch
from ultralytics import YOLO
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
yolo_model = YOLO("yolov8s-seg.pt")  # Use the segmentation model
caption_model_name = "nlpconnect/vit-gpt2-image-captioning"
caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model_name).to(device)
caption_processor = ViTImageProcessor.from_pretrained(caption_model_name)
caption_tokenizer = AutoTokenizer.from_pretrained(caption_model_name)

def extract_objects(image: Image.Image):
    """ Extracts segmented objects using YOLOv8 segmentation. """
    results = yolo_model(image, task="segment", conf=0.25)  # Lower confidence threshold
    object_names = []
    if results[0].masks is not None:
        for box in results[0].boxes:
            object_names.append(results[0].names[int(box.cls)])
    return list(set(object_names)), results

def generate_caption(image: Image.Image, detected_objects: list) -> str:
    """ Generates an enriched caption incorporating detected objects. """
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values.to(device)
    output_ids = caption_model.generate(pixel_values, max_length=40)
    base_caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    obj_text = f" Detected objects: {', '.join(detected_objects)}." if detected_objects else ""
    return base_caption + obj_text

def save_segmented_result_image(results, image_path):
    """ Saves an image with segmentation masks drawn over it. """
    seg_img_array = results[0].plot()  # YOLO's plot() method returns a numpy array
    seg_image = Image.fromarray(seg_img_array)
    
    seg_folder = "segmented_results"
    os.makedirs(seg_folder, exist_ok=True)
    seg_img_path = os.path.join(seg_folder, f"seg_{os.path.basename(image_path)}")
    seg_image.save(seg_img_path)
    print(f"[✓] Saved segmented image at {seg_img_path}")

def caption_images_from_folder(root_folder: str, output_csv: str = "captions.csv"):
    """ Processes images from a folder and saves segmentation overlay images. """
    # Get subfolders (use the first subfolder for processing)
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print("No subfolders found.")
        return

    image_files = [f for f in os.listdir(subfolders[0]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    captions = []

    for img_file in image_files:
        img_path = os.path.join(subfolders[0], img_file)
        try:
            # Open and resize image for YOLO
            image = Image.open(img_path).convert("RGB").resize((640, 480))
            objects, results = extract_objects(image)
            caption = generate_caption(image, objects)
            captions.append({"image": img_file, "caption": caption})
            print(f"[✓] {img_file}: {caption}")
            
            # Save the segmentation overlay image to check segmentation quality
            save_segmented_result_image(results, img_path)
        except Exception as e:
            print(f"[✗] Failed on {img_file}: {e}")

    pd.DataFrame(captions).to_csv(output_csv, index=False)
    print(f"\nSaved {len(captions)} captions to {output_csv}")

if __name__ == "__main__":
    caption_images_from_folder("images")
