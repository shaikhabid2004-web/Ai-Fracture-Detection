import os
import sys
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from collections import defaultdict
import random

def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(image)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def prepare_fracatlas_dataset(
    raw_root=r"C:\Users\shaik\OneDrive\Desktop\xray\xrayy kaggle\FracAtlas",
    out_root=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\fracatlas_prepared",
    target_size=(256, 256),
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_root, exist_ok=True)
    
    img_dir_frac = os.path.join(raw_root, "images", "Fractured")
    img_dir_non = os.path.join(raw_root, "images", "Non_fractured")
    voc_dir = os.path.join(raw_root, "Annotations", "PASCAL VOC")
    split_dir = os.path.join(raw_root, "Utilities", "Fracture Split")
    dataset_csv = os.path.join(raw_root, "dataset.csv")
    
    print("=" * 80)
    print("      STEP 1: SCANNING & DEDUPLICATION OF FRACATLAS DATASET")
    print("=" * 80)
    
    # 1. Scan MD5 and perceptual hashes for exact & near duplicates
    all_files = []
    if os.path.exists(img_dir_frac):
        for f in os.listdir(img_dir_frac):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append((f, "fractured", os.path.join(img_dir_frac, f)))
    if os.path.exists(img_dir_non):
        for f in os.listdir(img_dir_non):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append((f, "normal", os.path.join(img_dir_non, f)))
                
    print(f"Total raw images located: {len(all_files)}")
    
    seen_md5 = {}
    seen_dhash = {}
    duplicates = []
    unique_files = []
    
    for fname, label, fpath in all_files:
        with open(fpath, "rb") as fp:
            data = fp.read()
            h_md5 = hashlib.md5(data).hexdigest()
            
        if h_md5 in seen_md5:
            duplicates.append((fname, fpath, f"Exact MD5 match with {seen_md5[h_md5][0]}"))
            continue
            
        try:
            with Image.open(fpath) as pil_img:
                h_dhash = dhash(pil_img)
        except Exception:
            h_dhash = None
            
        if h_dhash is not None and h_dhash in seen_dhash:
            duplicates.append((fname, fpath, f"Perceptual dHash match with {seen_dhash[h_dhash][0]}"))
            continue
            
        seen_md5[h_md5] = (fname, fpath)
        if h_dhash is not None:
            seen_dhash[h_dhash] = (fname, fpath)
        unique_files.append((fname, label, fpath))
        
    print(f"Purged {len(duplicates)} duplicate files.")
    print(f"Remaining unique, high-quality images: {len(unique_files)}")
    
    # 2. Read dataset.csv metadata
    meta_by_id = {}
    if os.path.exists(dataset_csv):
        with open(dataset_csv, "r", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                meta_by_id[row["image_id"]] = row
                
    # 3. Read official split IDs for fractures
    official_train = set()
    official_val = set()
    official_test = set()
    
    if os.path.exists(os.path.join(split_dir, "train.csv")):
        with open(os.path.join(split_dir, "train.csv"), "r", encoding="utf-8") as fp:
            official_train = {r["image_id"] for r in csv.DictReader(fp)}
    if os.path.exists(os.path.join(split_dir, "valid.csv")):
        with open(os.path.join(split_dir, "valid.csv"), "r", encoding="utf-8") as fp:
            official_val = {r["image_id"] for r in csv.DictReader(fp)}
    if os.path.exists(os.path.join(split_dir, "test.csv")):
        with open(os.path.join(split_dir, "test.csv"), "r", encoding="utf-8") as fp:
            official_test = {r["image_id"] for r in csv.DictReader(fp)}
            
    print(f"Official FracAtlas fracture splits: Train={len(official_train)}, Val={len(official_val)}, Test={len(official_test)}")
    
    # 4. Stratify into balanced Train/Val/Test
    # Separate unique fractured and normal
    unique_frac = [item for item in unique_files if item[1] == "fractured"]
    unique_norm = [item for item in unique_files if item[1] == "normal"]
    
    print(f"Available Unique Fractures: {len(unique_frac)} | Unique Normals: {len(unique_norm)}")
    
    # Assign fractures according to official split or 80/10/10
    frac_train, frac_val, frac_test = [], [], []
    for item in unique_frac:
        fname = item[0]
        if fname in official_train:
            frac_train.append(item)
        elif fname in official_val:
            frac_val.append(item)
        elif fname in official_test:
            frac_test.append(item)
        else:
            # Fallback random
            r = random.random()
            if r < 0.80:
                frac_train.append(item)
            elif r < 0.90:
                frac_val.append(item)
            else:
                frac_test.append(item)
                
    # Sample balanced normals (1:1 ratio matching fracture counts)
    random.shuffle(unique_norm)
    n_train_target = len(frac_train)
    n_val_target = len(frac_val)
    n_test_target = len(frac_test)
    
    norm_train = unique_norm[:n_train_target]
    norm_val = unique_norm[n_train_target : n_train_target + n_val_target]
    norm_test = unique_norm[n_train_target + n_val_target : n_train_target + n_val_target + n_test_target]
    
    splits = {
        "train": {"fracture": frac_train, "normal": norm_train},
        "val": {"fracture": frac_val, "normal": norm_val},
        "test": {"fracture": frac_test, "normal": norm_test}
    }
    
    print("\nBalanced Stratified Distribution:")
    for split_name, categories in splits.items():
        n_f = len(categories["fracture"])
        n_n = len(categories["normal"])
        print(f"  [{split_name.upper():5s}] -> Fractures: {n_f:3d} | Normals: {n_n:3d} | Total: {n_f + n_n:3d}")
        
    print("\n=" * 80)
    print("      STEP 2: PRE-RESIZING & CACHING TO HIGH-SPEED 256x256 DATASET")
    print("=" * 80)
    
    annotations = {}
    manifest_rows = []
    
    for split_name, categories in splits.items():
        for cat_label, file_list in categories.items():
            dest_dir = os.path.join(out_root, split_name, cat_label)
            os.makedirs(dest_dir, exist_ok=True)
            
            for fname, label_type, fpath in file_list:
                dest_fname = os.path.splitext(fname)[0] + ".png"
                dest_path = os.path.join(dest_dir, dest_fname)
                
                try:
                    with Image.open(fpath) as img:
                        # Convert to grayscale and resize smoothly with bilinear interpolation
                        img_gray = img.convert("L")
                        w_orig, h_orig = img_gray.size
                        res_img = img_gray.resize(target_size, Image.Resampling.BILINEAR)
                        res_img.save(dest_path, "PNG", optimize=True)
                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    continue
                    
                # Parse PASCAL VOC XML if fracture
                bboxes = []
                xml_path = os.path.join(voc_dir, os.path.splitext(fname)[0] + ".xml")
                if os.path.exists(xml_path):
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        for obj in root.findall("object"):
                            bndbox = obj.find("bndbox")
                            if bndbox is not None:
                                xmin = float(bndbox.find("xmin").text)
                                ymin = float(bndbox.find("ymin").text)
                                xmax = float(bndbox.find("xmax").text)
                                ymax = float(bndbox.find("ymax").text)
                                # Scale to target 256x256
                                scale_x = target_size[0] / w_orig
                                scale_y = target_size[1] / h_orig
                                bboxes.append([
                                    int(round(xmin * scale_x)),
                                    int(round(ymin * scale_y)),
                                    int(round(xmax * scale_x)),
                                    int(round(ymax * scale_y))
                                ])
                    except Exception:
                        pass
                        
                meta = meta_by_id.get(fname, {})
                body_part = "mixed"
                for part in ["hand", "leg", "hip", "shoulder"]:
                    if meta.get(part) == "1":
                        body_part = part
                        break
                        
                entry = {
                    "filename": dest_fname,
                    "original_filename": fname,
                    "split": split_name,
                    "label": 1 if cat_label == "fracture" else 0,
                    "category": cat_label,
                    "body_part": body_part,
                    "hardware": meta.get("hardware", "0") == "1",
                    "bboxes_256": bboxes,
                    "rel_path": os.path.join(split_name, cat_label, dest_fname)
                }
                annotations[dest_fname] = entry
                manifest_rows.append(entry)
                
    # Save annotations JSON and CSV manifest
    with open(os.path.join(out_root, "annotations.json"), "w", encoding="utf-8") as fp:
        json.dump(annotations, fp, indent=2)
        
    manifest_csv_path = os.path.join(out_root, "manifest.csv")
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["filename", "original_filename", "split", "label", "category", "body_part", "hardware", "rel_path"])
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow({
                "filename": r["filename"],
                "original_filename": r["original_filename"],
                "split": r["split"],
                "label": r["label"],
                "category": r["category"],
                "body_part": r["body_part"],
                "hardware": r["hardware"],
                "rel_path": r["rel_path"]
            })
            
    print(f"\n[SUCCESS] FracAtlas dataset prepared at: {out_root}")
    print(f"Total balanced dataset: {len(manifest_rows)} images cached.")
    return out_root

if __name__ == "__main__":
    prepare_fracatlas_dataset()
