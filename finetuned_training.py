from finetuned_model_loader import blip_model, blip_processor, device
from config import *
from finetuned_annotation_parser import parse_annotations, remove_determined_values
from finetuned_caption_refiner import refine_caption
from finetuned_image_utils import extract_combined_mask, run_sam_full_image
from finetuned_ocr_utils import extract_text
from PIL import Image
import torch
import os
import json
import numpy as np
from transformers import get_linear_schedule_with_warmup
import time
import csv

# Extract paths from config and define device
captions_path = CAPTIONS_PATH
captions_path_val = CAPTIONS_PATH_VAL
segmented_images_path = SEGMENTED_IMAGES_PATH

def training_loop(root_folder: str = "images"):
    annotations_path = os.path.join(root_folder, "annotations.md")
    if not os.path.exists(annotations_path):
        print("annotations.md not found in the images folder.")
        return

    # Parse the original annotations and prepare targets
    original_annotations = parse_annotations(annotations_path)
    # Use the annotations without "Determined Values" as target captions.
    for key in original_annotations:
        original_annotations[key] = remove_determined_values(original_annotations[key])
        # print(f"Key: {key}, orifinal annotation: {original_annotations}")
    folder_ids = list(original_annotations.keys())
    if not folder_ids:
        print("No valid folder IDs found in annotations.md")
        return
    
    if os.path.exists(captions_path):
        with open(captions_path, "r") as f:
            refined_captions = json.load(f)
    else:
        refined_captions = {}
        print("Captions not found. They will be generated during the first epoch")
    
    if os.path.exists(captions_path_val):
        with open(captions_path_val, "r") as f:
            refined_captions_val = json.load(f)
    else:
        refined_captions_val = {}
        print("Captions for val set not found. They will be generated during the first epoch")

    if not os.path.exists(segmented_images_path):
        os.makedirs(segmented_images_path, exist_ok=True)
        # Generate segmented images if subfolders don't exist
        for folder in os.listdir(root_folder):
            folder_path = os.path.join(root_folder, folder)
            segmented_folder_path = os.path.join(segmented_images_path, folder)
            
            os.makedirs(segmented_folder_path, exist_ok=True)
            print(f"Generating segmentations for folder: {folder}")
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    image = Image.open(image_path).convert("RGB").resize((640, 480))
                    combined_segment = extract_combined_mask(image)

                    if combined_segment:
                        name, ext = os.path.splitext(filename)
                        new_filename = f"{name}_segmented{ext}"
                        seg_image_path = os.path.join(segmented_folder_path, new_filename)
                        combined_segment.save(seg_image_path)
                    else:
                        sam_segment = run_sam_full_image(image)
                        if sam_segment:
                            name, ext = os.path.splitext(filename)
                            new_filename = f"{name}_segmented{ext}"
                            seg_image_path = os.path.join(segmented_folder_path, new_filename)
                            sam_segment.save(seg_image_path)
                        else:
                            print(f"Segmentation failed for: {image_path}")

    # Determine available subfolders in the images folder that match the annotations.
    available_folders = []
    test_folders = [] 
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            if folder in folder_ids:
                available_folders.append(folder_path)
            else:
                test_folders.append(folder_path)

    available_folders.sort()

    if len(available_folders) < 3:
        print("Need at least 3 folders (for training, validation, and test).")
        return

    # Reserve one folder for validation and one for test; all others for training.
    validation_folder = available_folders[0]
    training_folders = available_folders[1:]

    print("Folder split:")
    print("  Training folders:", [os.path.basename(f) for f in training_folders])
    print("  Validation folder:", os.path.basename(validation_folder))
    print("  Test folder:", [os.path.basename(f) for f in test_folders])

    # Setup optimizer and scheduler to fine-tune BLIP (we update only the captioning model)
    num_epochs = 2
    optimizer = torch.optim.Adam(blip_model.parameters(), lr=1e-6)
    # Count total number of training steps (rough estimate)
    total_steps = sum([len(os.listdir(f)) for f in training_folders if os.listdir(f)]) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    # For collecting outputs (for final CSV)
    all_rows = []

    # Begin training epochs
    for epoch in range(1, num_epochs + 1):

        blip_model.train()
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        total_train_loss = 0.0
        train_image_count = 0

        # Gather all training images from all folders
        training_samples = []
        for folder in training_folders:
            folder_name = os.path.basename(folder)
            target_caption = original_annotations.get(folder_name)
            if target_caption is None:
                continue
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder, filename)
                    training_samples.append((image_path, folder_name, target_caption))

        # Add segmented images
        segmented_folder = os.path.join(segmented_images_path, folder_name)
        if os.path.exists(segmented_folder):
            for filename in os.listdir(segmented_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    seg_image_path = os.path.join(segmented_folder, filename)
                    training_samples.append((seg_image_path, folder_name, target_caption))

        # Shuffle training samples
        np.random.shuffle(training_samples)
        # Train on each image independently
        for image_index, (image_path, folder_name, target_caption) in enumerate(training_samples, start=1):
            print(f"\n  Training image {image_index}/{len(training_samples)} from folder: {folder_name} - {os.path.basename(image_path)}")
            try:
                image = Image.open(image_path).convert("RGB").resize((640, 480))
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            # Step 1: Generate caption
            blip_model.eval()
            with torch.no_grad():
                inference_inputs = blip_processor(images=image, return_tensors="pt").to(device)
                generated_ids = blip_model.generate(**inference_inputs)
                predicted_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            blip_model.train()

            print(f"    Generated caption: {predicted_caption}")

            # Step 2: Refine with OpenAI
            if image_path in refined_captions:
                refined_caption = refined_captions[image_path]
                print(f"    Using cached refined caption: {refined_caption}")
            else:
                try:
                    start_time = time.time()
                    refined_caption = refine_caption(predicted_caption, target_caption)
                    print(f"    Refined caption (in {time.time() - start_time:.2f}s): {refined_caption}")

                except Exception as e:
                    print(f"Refinement error: {e}")
                    refined_caption = predicted_caption

            # === Step 3: Use the refined caption for training ===
            inputs = blip_processor(
                images=image,
                text=refined_caption,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            labels[labels == blip_processor.tokenizer.pad_token_id] = -100

            outputs = blip_model(**inputs, labels=labels)
            loss = outputs.loss

            total_loss = loss

            # Backprop and step
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            train_image_count += 1

            refined_captions[image_path] = refined_caption


            # === Step 4: Log all results ===
            all_rows.append([
                image_path,
                folder_name,
                target_caption,
                predicted_caption,
                refined_caption
            ])

        avg_train_loss = total_train_loss / train_image_count if train_image_count else 0.0
        print(f"Epoch {epoch} completed. Average training loss: {avg_train_loss:.4f}")

        # Save the fine-tuned model checkpoint.
        model_save_path = os.path.join(root_folder, "blip_finetuned.pt")
        torch.save(blip_model.state_dict(), model_save_path)
        print(f"Fine-tuned BLIP model saved to {model_save_path}.")

        # ---- Validation Step (evaluate on the reserved validation folder) ---- #
        blip_model.eval()
        val_loss = 0.0
        val_image_count = 0
        validation_captions = []

        val_folder_name = os.path.basename(validation_folder)
        target_caption_val = original_annotations.get(val_folder_name, "No Target")

        image_files = [f for f in os.listdir(validation_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        segmented_val_folder = os.path.join(segmented_images_path, val_folder_name)
        if os.path.exists(segmented_val_folder):
            segmented_image_files = [
                os.path.join(segmented_val_folder, f)
                for f in os.listdir(segmented_val_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        image_files.extend(segmented_image_files)

        total_val_images = len(image_files)
        print(f"\nProcessing validation folder '{val_folder_name}' with {total_val_images} images.")

        for val_index, filename in enumerate(image_files, start=1):
            image_path = os.path.join(validation_folder, filename)
            print(f"  Validation: processing image {val_index}/{total_val_images}: {filename}")
            try:
                image = Image.open(image_path).convert("RGB").resize((640, 480))
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            # === Step 1: Generate caption with BLIP ===
            with torch.no_grad():
                inference_inputs = blip_processor(images=image, return_tensors="pt").to(device)
                generated_ids = blip_model.generate(**inference_inputs)
                predicted_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            print(f"    Generated caption: {predicted_caption}")

            # Step 2: Refine with OpenAI
            if image_path in refined_captions_val:
                refined_caption = refined_captions_val[image_path]
                print(f"    Using cached refined caption: {refined_caption}")
            else:
                try:
                    start_time = time.time()
                    refined_caption = refine_caption(predicted_caption, target_caption_val)
                    print(f"    Refined caption (in {time.time() - start_time:.2f}s): {refined_caption}")

                except Exception as e:
                    print(f"Refinement error: {e}")
                    refined_caption = predicted_caption

            # === Step 3: Compute validation loss using refined caption ===
            inputs = blip_processor(
                images=image,
                text=refined_caption,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                labels = inputs["input_ids"].clone()
                labels[labels == blip_processor.tokenizer.pad_token_id] = -100
                outputs = blip_model(**inputs, labels=labels)
                loss = outputs.loss

            val_loss += loss.item()
            val_image_count += 1

            # === Step 4: Log result ===
            validation_captions.append((image_path, predicted_caption, refined_caption))

            refined_captions_val[image_path] = refined_caption

        avg_val_loss = val_loss / val_image_count if val_image_count else 0.0
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Save refined captions after epoch 1
        if epoch == 1:
            if not os.path.exists(captions_path):
                try:
                    with open(captions_path, 'w', encoding='utf-8') as f:
                        json.dump(refined_captions, f, indent=2, ensure_ascii=False)
                    print(f"Saved refined training captions to {captions_path}")
                except Exception as e:
                    print(f"Failed to save refined training captions: {e}")
            if not os.path.exists(captions_path_val):
                try:
                    with open(captions_path_val, 'w', encoding='utf-8') as f:
                        json.dump(refined_captions_val, f, indent=2, ensure_ascii=False)
                    print(f"Saved refined validation captions to {captions_path_val}")
                except Exception as e:
                    print(f"Failed to save refined validation captions: {e}")

    # ---- Final Test Evaluation on the Test Folders ---- #
    
    # Loop through all test folders
    for test_folder in [validation_folder]:
        print(f"\n--- Final Evaluation on Test Folder: {os.path.basename(test_folder)} ---")
        test_folder_name = os.path.basename(test_folder)

        image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        segmented_test_folder = os.path.join(segmented_images_path, test_folder_name)
        if os.path.exists(segmented_test_folder):
            segmented_image_files = [
                os.path.join(segmented_test_folder, f)
                for f in os.listdir(segmented_test_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        image_files.extend(segmented_image_files)

        total_test_images = len(image_files)
        print(f"\nProcessing test folder '{test_folder_name}' with {total_test_images} images.")

        rows = [["image_path", "caption", "text_detected"]]

        for test_index, filename in enumerate(image_files, start=1):
            image_path = os.path.join(test_folder, filename)
            print(f"  Test: processing image {test_index}/{total_test_images}: {filename}")
            try:
                image = Image.open(image_path).convert("RGB").resize((640, 480))
            except Exception as e:
                print(f"Failed to open {image_path}: {e}")
                continue

            with torch.no_grad():
                inference_inputs = blip_processor(images=image, return_tensors="pt").to(device)
                generated_ids = blip_model.generate(**inference_inputs)
                predicted_caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

            detected_text = extract_text(image_path)

            rows.append([
                image_path, predicted_caption, detected_text
            ])

        print(f"Processed {len(image_files)} test images.")

        # ---- Save Final CSV Output for this folder ---- #
        output_csv = f"evaluation_{test_folder_name}.csv"
        with open(output_csv, mode="w", newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

        print(f"\nFinished! Results saved to {output_csv}")