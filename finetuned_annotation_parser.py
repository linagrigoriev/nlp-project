import re

def get_training_ids(annotations_path):
    training_ids = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                training_ids.append(line[3:].strip())
    return training_ids

def parse_annotations(annotations_path):
    annotations = {}
    current_folder = None
    current_lines = []
    with open(annotations_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("## "):
                if current_folder:
                    annotations[current_folder] = "\n".join(current_lines)
                current_folder = line[3:].strip()
                current_lines = [line.strip()]
            else:
                current_lines.append(line.strip())
        if current_folder:
            annotations[current_folder] = "\n".join(current_lines)
    return annotations

def remove_determined_values(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    skip = False
    for line in lines:
        if line.startswith("### Determined Values"):
            skip = True
            continue
        if skip and line.startswith("### "):
            skip = False
        if skip or not line.strip():
            continue
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"^- ", "", line)
        line = re.sub(r"###", "", line).strip()
        if any(keyword in line.lower() for keyword in [
            "vehicle", "truck", "car", "machine", "container", "unit", "equipment",
            "damaged", "intact", "burnt", "destroyed", "missing", "functional", "partial"
        ]):
            cleaned_lines.append(line)
    return " ".join(cleaned_lines)
