import re
from openai_client import client
from config import MODEL

def clean_duplicate_phrases(text: str) -> str:
    phrases = re.split(r",| and ", text)
    seen = set()
    cleaned_phrases = []
    for phrase in phrases:
        norm = phrase.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            cleaned_phrases.append(phrase.strip())
    return ", ".join(cleaned_phrases)

def remove_repeated_parts(text: str) -> str:
    return re.sub(r",+\s*", ", ", text).strip(",. ")

def refine_caption(image_caption: str, annotation_summary: str):
    image_caption = remove_repeated_parts(image_caption)
    cleaned_caption = clean_duplicate_phrases(image_caption)

    prompt = f"""
You are an expert reviewer of machine-generated image captions. Your task is to:

- Verify if any damage, defects, or specific states (e.g., flat tire, broken windshield) mentioned in the caption are clearly supported by the visible evidence or the annotation summary.
- If such claims are NOT supported, remove or correct them.
- Keep the caption concise and truthful based only on visible evidence or annotations.
- Avoid adding any new technical details, brand names, or assumptions.
- Neutral, descriptive details about visible components (e.g., GPS device, car window, dashboard) or tools are allowed and should be preserved.
- Do NOT mention any colors in your caption. Ignore color information entirely.
- Ignore black squares or masked areas added to hide sensitive data; do not mention them explicitly.
- If the caption mentions a piece of paper, document, plates or similar item, do NOT reinterpret it as anything else—accept it as is.
- Maintain natural, clear language.

Original Caption:
"{cleaned_caption}"

Annotation Summary:
"{annotation_summary}"

Based on the above, provide the corrected and verified caption only.
"""

    print("Sending request to OpenAI API...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You carefully verify and refine captions to ensure factual correctness and avoid hallucinated defects or brands."
            },
            {"role": "user", "content": prompt}
        ]
    )

    refined = response.choices[0].message.content.strip()
    refined = remove_repeated_parts(refined)
    return clean_duplicate_phrases(refined)
