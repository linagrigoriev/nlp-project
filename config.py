import os
from dotenv import load_dotenv
import torch

load_dotenv()

# Paths
FINETUNED_PATH = os.path.join("output", "finetuned.pt")
CAPTIONS_PATH = os.path.join("output", "ground_truth_captions.json")
CAPTIONS_PATH_VAL = os.path.join("output", "ground_truth_captions_val.json")
ANNOTATIONS_MD = os.path.join("images", "annotations.md")
SEGMENTED_IMAGES_PATH = os.path.join("output", "segmented_images")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenAI API
API_KEY = os.getenv("OPENAI_API_KEY_R")
BASE_URL = os.getenv("OPENAI_BASE_URL_R")
MODEL = os.getenv("OPENAI_MODEL_R")