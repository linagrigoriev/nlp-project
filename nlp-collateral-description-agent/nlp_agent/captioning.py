import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class CaptionGenerator:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

        if model_path:
            state_dict = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large") #change to base if fine-tuned BLIP is used



    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=50)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)