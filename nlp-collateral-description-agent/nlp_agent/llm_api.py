import re
import requests

API_KEY = ""  #your openai api key
URL = "https://api.openai.com/v1/chat/completions" #adjust if you're not using openai
MODEL_ID = "gpt-4-turbo"

SYSTEM_PROMPT = """You are a professional vehicle evaluator. Given the following captions and texts extracted from images, some of which may be incomplete, corrupted, or partially lost, analyze the content carefully.

Produce a detailed, structured markdown report with the following sections:

### Identification & General Data
- Asset Type
- Manufacturer
- Model
- Vehicle Type
- Year of Manufacture
- First Registration Date
- First Registration Country
- Engine Power
- Engine Displacement
- Environmental Classification
- Seating Capacity
- Transmission Type
- Technical Inspection Valid Until
- Odometer Reading
- Accepted Mileage
- Number of Keys
- Service Book
- Usage
- Document Date

### Inspection Methods

### Condition Assessment

### Valuation Principles

### Documentation & Accessories

Input captions (from images) and related texts:
{captions_text}

Use the extracted captions and text to fill in all fields as completely as possible.
- If exact values are missing, try to infer or deduce plausible values from related text fragments, typical vehicle specs, or known domain standards.
- Map codes or abbreviations to their meaning when possible (e.g., "SEBESSÉGVÁLTÓ FAJTÁJA (KÓDSZÁMA): 2" means Automatic transmission).
- If the exact date or number is partial or ambiguous, provide the most likely interpretation with a note that it was deduced.
- Include uncertainty remarks where deduction was applied.      
- Ensure the report is in clear markdown format with the sections and bullet points as outlined.  
- Keep the report professional, detailed, and concise.
"""

CLEAN_SPACES = re.compile(r"[ \t]+")


def _norm(text: str) -> str:
    return CLEAN_SPACES.sub(" ", text.strip())


def generate_structured_description(caption: str,
                                    ocr_text: str,
                                    extra: str = "",
                                    *,
                                    temperature: float = 0.3,
                                    max_tokens: int = 1024,
                                    api_key: str = API_KEY) -> str | None:
    """Return English markdown report text; None on failure."""
    user_prompt_parts = [
        "\n\n### IMAGE CAPTIONS\n", _norm(caption),
        "\n\n### OCR TEXT (cleaned)\n", _norm(ocr_text)
    ]
    if extra:
        user_prompt_parts.extend(["\n\n### EXTRA CONTEXT\n", _norm(extra)])

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "".join(user_prompt_parts)}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(
            URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=90
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print("OpenAI API error:", exc)
        return None
