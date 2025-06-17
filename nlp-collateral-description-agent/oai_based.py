import openai
import base64
import os

client = openai.OpenAI(api_key="")

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def generate_caption_and_text(image_path):
    base64_image = image_to_base64(image_path)

    prompt = (
        "You are an expert automotive appraiser. Carefully analyze the image and describe:\n"
        "1. The specific vehicle part visible and its condition (e.g. intact, damaged, worn, dirty, rusted).\n"
        "3. Any visible issues (scratches, dents, rust, missing parts, etc.).\n"
        "4. If readable, extract and include any visible text (labels, stickers, serial numbers).\n\n"
        "Be concise and factual. Avoid speculation. Format the output clearly using bullet points or sections."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()

def process_all_images(input_folder, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(input_folder, filename)
                print(f"Processing {filename}...")

                try:
                    result = generate_caption_and_text(image_path)

                    f.write(f"=== {filename} ===\n")
                    f.write(result + "\n\n")
                    print(f"✅ Processed and saved: {filename}")

                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = "/images/folder_name"
    output_file = "/descriptions.txt"
    process_all_images(input_folder, output_file)
