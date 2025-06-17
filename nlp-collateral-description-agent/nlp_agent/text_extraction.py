import cv2
import torch
import easyocr

class TextExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en', 'hu'], gpu=torch.cuda.is_available())

    def preprocess(self, img):
        height, width, _ = img.shape
        scale_factor = 1.2  # fixed small upscale
        img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
        return img


    def extract_text(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return "Image not found or cannot be read."

        processed_img = self.preprocess(img)
        results = self.reader.readtext(processed_img)

        detected_texts = [res[1] for res in results]
        return ", ".join(detected_texts) if detected_texts else ""

