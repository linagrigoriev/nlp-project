import os
from PIL import Image
import torch
from nlp_agent.segmentation import Segmenter
from nlp_agent.captioning import CaptionGenerator
from nlp_agent.text_extraction import TextExtractor
from nlp_agent.llm_api import generate_structured_description
from nlp_agent.valuation import extract_fields, estimate_value, update_report_with_valuation_text


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmenter = Segmenter(
    yolo_path="path_to_yolo/models/yolov8x-seg.pt", 
    sam_checkpoint="path_to_sam/models/sam_vit_l_0b3195.pth"
)
text_extractor = TextExtractor()

captioner = CaptionGenerator(None)
# model_path="/models/blip_finetuned.pt"  # to use fine-tuned BLIP

def process_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((640, 480))
    detected_text = text_extractor.extract_text(image_path)

    # Try YOLO-based combined mask first
    combined_segment = segmenter.extract_combined_mask(image)

    # If YOLO segmentation fails, fallback to SAM full image segmentation
    if combined_segment is None:
        combined_segment = segmenter.run_sam_full_image(image)

    if combined_segment is not None:
        caption = captioner.generate_caption(combined_segment)
    else:
        caption = "Segmentation failed."

    enriched_caption = f"{caption}. Text found: {detected_text}"
    return enriched_caption


if __name__ == "__main__":
    folder_path = "path_to_images/images/157515_v2"
    captions_file = "captions.txt"
    report_file = "asset_appraisal_report.md"

    supported_exts = {".jpg", ".jpeg", ".png"}

    # Step 1: Process images and write captions
    with open(captions_file, "w", encoding="utf-8") as f:
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_exts:
                continue

            image_path = os.path.join(folder_path, filename)
            print(f"Processing {filename} ...")
            enriched_caption = process_image(image_path)

            f.write(f"{enriched_caption}\n")

    print(f"✅ Image processing complete. Captions saved to: {captions_file}")

    # Step 2: Load captions and split into caption/text parts
    with open(captions_file, "r", encoding="utf-8") as f:
        raw = f.read()

    captions = []
    ocr_texts = []

    for line in raw.strip().split("\n"):
        if "Text found:" in line:
            caption_part, ocr_part = line.split("Text found:", maxsplit=1)
            captions.append(caption_part.strip())
            ocr_texts.append(ocr_part.strip())
        else:
            captions.append(line.strip())  # fallback

    combined_caption = " ".join(captions)
    combined_ocr = ", ".join(ocr_texts)
    extra_context = "- **Estimated average price:** "

    # Step 3: Generate report
    report = generate_structured_description(
        caption=combined_caption,
        ocr_text=combined_ocr,
        extra=extra_context,
        api_key=""  # Set your API key here
    )

    if report:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ Report generated and saved to: {report_file}")
    else:
        print("❌ Failed to generate the report.")

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_md = f.read()

        print("\n🔍 Extracting fields from report for valuation...")
        fields = extract_fields(report_md)

        if not fields.get("make") or not fields.get("model"):
            print("Missing 'make' or 'model' in the report. Skipping valuation.")
            exit(1)

        print("Estimating value from online sources...")
        estimate = estimate_value(
            fields.get("asset_type"),
            fields.get("make"),
            fields.get("model"),
            fields.get("year"),
            fields.get("mileage")
        )

        if not estimate["result"]:
            print("No price data found during search.")
            exit(1)

        avg_huf = estimate["result"]["approx_huf"]
        updated_report = update_report_with_valuation_text(report_md, avg_huf)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(updated_report)
        print(f"✅ Valuation info added to report: {avg_huf:,} HUF")

    except Exception as e:
        print(f"❌ Error updating report with valuation: {e}")
