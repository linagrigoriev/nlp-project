import cv2
from model_loader import ocr_reader

def extract_text(image_path):
    img = cv2.imread(image_path)
    results = ocr_reader.readtext(img)
    detected_texts = [res[1] for res in results]
    return ", ".join(detected_texts) if detected_texts else "No text detected"
