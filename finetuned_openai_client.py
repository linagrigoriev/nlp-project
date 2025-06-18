from openai import OpenAI
from config import API_KEY, BASE_URL

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)