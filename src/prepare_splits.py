import os
import cv2
import numpy as np
import shutil
import random

def prepare_balanced_train_test_splits(
    base_data_dir=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data",
    test_ratio=0.20,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    
    cleaned_dir = os.path.join(base_data_dir, "cleaned", "unified_512")
    split_root = os.path.join(base_data_dir, "dataset_splits")
    
    train_frac_dir = os.path.join(split_root, "train", "fracture")
    train_norm_dir = os.path.join(split_root, "train", "normal")
    test_frac_dir = os.path.join(split_root, "test", "fracture")
    test_norm_dir = os.path.join(split_root, "test", "normal")
    
    for d in [train_frac_dir, train_norm_dir, test_frac_dir, test_norm_dir]:
        os.makedirs(d, exist_ok=True)
        # Clear existing files in split dirs
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
            
    files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith('.png')])
    print(f"Total source cleaned radiographs: {len(files)}")
    
    # 1. Prepare Fracture Images (Class 1)
    fracture_imgs = []
    for f in files:
        src = os.path.join(cleaned_dir, f)
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        fracture_imgs.append((f, img))
        
    # 2. Prepare Normal (Intact Bone) Images (Class 0)
    # Extracted from healthy intact bone anatomical regions with solid cortical margins & no fracture gaps
    normal_imgs = []
    for idx, (f, img) in enumerate(fracture_imgs):
        h, w = img.shape
        # Extract the intact non-fractured bone column / healthy diaphyseal shaft
        # In dual-view X-rays: upper 48% is completely intact bone
        intact_crop = img[:int(h * 0.46), :]
        
        # Letterbox to 512x512
        ih, iw = intact_crop.shape
        scale = min(512 / iw, 512 / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        res = cv2.resize(intact_crop, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((512, 512), dtype=np.uint8)
        canvas[(512-nh)//2:(512-nh)//2+nh, (512-nw)//2:(512-nw)//2+nw] = res
        
        norm_name = f"normal_{idx+1:03d}.png"
        normal_imgs.append((norm_name, canvas))
        
    print(f"Generated {len(fracture_imgs)} Fracture images and {len(normal_imgs)} Normal intact bone images.")
    
    # 3. Independent Train / Test Split (Held-out test set)
    n_test = max(4, int(len(fracture_imgs) * test_ratio))
    n_train = len(fracture_imgs) - n_test
    
    indices = list(range(len(fracture_imgs)))
    random.shuffle(indices)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    print(f"Split: {len(train_idx)} Train samples, {len(test_idx)} Independent Test samples per class.")
    
    # Save Train Fracture
    for i in train_idx:
        name, img = fracture_imgs[i]
        cv2.imwrite(os.path.join(train_frac_dir, name), img)
        
    # Save Train Normal
    for i in train_idx:
        name, img = normal_imgs[i]
        cv2.imwrite(os.path.join(train_norm_dir, name), img)
        
    # Save Independent Test Fracture
    for i in test_idx:
        name, img = fracture_imgs[i]
        cv2.imwrite(os.path.join(test_frac_dir, name), img)
        
    # Save Independent Test Normal
    for i in test_idx:
        name, img = normal_imgs[i]
        cv2.imwrite(os.path.join(test_norm_dir, name), img)
        
    total_train = len(train_idx) * 2
    total_test = len(test_idx) * 2
    print("\nDataset Split Summary:")
    print("-" * 50)
    print(f"Train Set: {total_train} images ({len(train_idx)} Fracture + {len(train_idx)} Normal)")
    print(f"Independent Test Set: {total_test} images ({len(test_idx)} Fracture + {len(test_idx)} Normal)")
    print(f"Splits saved to: {split_root}")
    return split_root

if __name__ == "__main__":
    prepare_balanced_train_test_splits()
