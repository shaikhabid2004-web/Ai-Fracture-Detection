"""
MURA Dataset Integrator & Preprocessor for BoneSense AI
Uses standard Python libraries (no pandas needed).
Scans MURA-v1.1, extracts curated positive (fracture) and negative (normal) cases
from XR_WRIST, XR_ELBOW, XR_HAND, XR_SHOULDER and places them into training splits.
"""

import os
import sys
import glob
import shutil

WORKSPACE_ROOT = r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti"
DEST_SPLIT_TRAIN_FRAC = os.path.join(WORKSPACE_ROOT, "data", "dataset_splits", "train", "fracture")
DEST_SPLIT_TRAIN_NORM = os.path.join(WORKSPACE_ROOT, "data", "dataset_splits", "train", "normal")
MURA_DIR = os.path.join(WORKSPACE_ROOT, "data", "MURA-v1.1")

def integrate_mura(max_samples_per_part=50, target_parts=["XR_WRIST", "XR_ELBOW", "XR_HAND", "XR_SHOULDER"]):
    print(f"[MURA Integrator] Starting integration from {MURA_DIR}...")
    
    if not os.path.exists(MURA_DIR):
        print(f"[MURA Integrator] Error: {MURA_DIR} not found.")
        return False

    train_dir = os.path.join(MURA_DIR, "train")
    if not os.path.exists(train_dir):
        print(f"[MURA Integrator] Error: {train_dir} not found.")
        return False

    os.makedirs(DEST_SPLIT_TRAIN_FRAC, exist_ok=True)
    os.makedirs(DEST_SPLIT_TRAIN_NORM, exist_ok=True)

    total_added_pos = 0
    total_added_neg = 0

    for body_part in target_parts:
        part_dir = os.path.join(train_dir, body_part)
        if not os.path.exists(part_dir):
            print(f" -> Skipping {body_part} (folder not found)")
            continue

        part_pos = 0
        part_neg = 0
        print(f" -> Extracting curated cases from {body_part} ...")

        # Patient folders
        patients = sorted(os.listdir(part_dir))
        for patient in patients:
            patient_dir = os.path.join(part_dir, patient)
            if not os.path.isdir(patient_dir):
                continue

            for study in os.listdir(patient_dir):
                study_dir = os.path.join(patient_dir, study)
                if not os.path.isdir(study_dir):
                    continue

                is_positive = "positive" in study.lower()

                # Get images in study
                for img_file in os.listdir(study_dir):
                    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    src_img = os.path.join(study_dir, img_file)

                    if is_positive and part_pos < max_samples_per_part:
                        out_name = f"mura_{body_part.lower()}_pos_{patient}_{study}_{img_file}"
                        dst = os.path.join(DEST_SPLIT_TRAIN_FRAC, out_name)
                        shutil.copy2(src_img, dst)
                        part_pos += 1
                        total_added_pos += 1
                    elif not is_positive and part_neg < max_samples_per_part:
                        out_name = f"mura_{body_part.lower()}_neg_{patient}_{study}_{img_file}"
                        dst = os.path.join(DEST_SPLIT_TRAIN_NORM, out_name)
                        shutil.copy2(src_img, dst)
                        part_neg += 1
                        total_added_neg += 1

            if part_pos >= max_samples_per_part and part_neg >= max_samples_per_part:
                break

        print(f"    Completed {body_part}: +{part_pos} Fractures, +{part_neg} Normals")

    print(f"\n=======================================================")
    print(f"[MURA Integrator] INTEGRATION SUMMARY:")
    print(f"  Total Fracture (Positive) cases added: +{total_added_pos}")
    print(f"  Total Normal (Negative) cases added  : +{total_added_neg}")
    print(f"  Grand Total MURA X-rays integrated   : +{total_added_pos + total_added_neg}")
    print(f"=======================================================")
    return True

if __name__ == "__main__":
    integrate_mura()
