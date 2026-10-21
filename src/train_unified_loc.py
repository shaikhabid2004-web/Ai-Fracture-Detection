import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork, CrackLocalizationHead

def generate_gaussian_target(bboxes_256, target_size=64, img_size=256):
    target = np.zeros((target_size, target_size), dtype=np.float32)
    if not bboxes_256:
        return target
    
    scale = target_size / img_size
    y_coords, x_coords = np.mgrid[0:target_size, 0:target_size]
    
    for box in bboxes_256:
        xmin, ymin, xmax, ymax = box
        cx = (xmin + xmax) / 2.0 * scale
        cy = (ymin + ymax) / 2.0 * scale
        w = max(4.0, (xmax - xmin) * scale)
        h = max(4.0, (ymax - ymin) * scale)
        
        sigma_x = max(1.5, w / 2.5)
        sigma_y = max(1.5, h / 2.5)
        
        gauss = np.exp(-(((x_coords - cx) ** 2) / (2 * sigma_x ** 2) + ((y_coords - cy) ** 2) / (2 * sigma_y ** 2)))
        target = np.maximum(target, gauss)
        
    return target

class FracAtlasUnifiedDataset(Dataset):
    def __init__(self, data_root, annotations, split="train"):
        self.samples = []
        split_dir = os.path.join(data_root, split)
        
        for cat, label_val in [("fracture", 1.0), ("normal", 0.0)]:
            cat_dir = os.path.join(split_dir, cat)
            if os.path.exists(cat_dir):
                for f in sorted(os.listdir(cat_dir)):
                    if f.endswith('.png'):
                        meta = annotations.get(f, {})
                        bboxes = meta.get("bboxes_256", [])
                        body_part = meta.get("body_part", "mixed")
                        
                        # Hard negative weighting: penalize normal paired-bone false alarms
                        if label_val == 0.0 and body_part in ["leg", "hand"]:
                            weight = 2.5 # Extra penalty for dual-bone false positives
                        else:
                            weight = 1.0
                            
                        self.samples.append((os.path.join(cat_dir, f), label_val, bboxes, f, weight, body_part))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        fpath, label_val, bboxes, fname, weight, body_part = self.samples[idx]
        img = Image.open(fpath).convert("L")
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_norm = (img_np - 0.5) / 0.5
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0)
        
        target_map = generate_gaussian_target(bboxes, target_size=64, img_size=256)
        target_tensor = torch.from_numpy(target_map).unsqueeze(0)
        
        return {
            "image": img_tensor,
            "label": torch.tensor(label_val, dtype=torch.float32),
            "target_map": target_tensor,
            "filename": fname,
            "weight": torch.tensor(weight, dtype=torch.float32),
            "body_part": body_part
        }

def compute_combined_loc_loss(pred_heat, targets):
    """
    Weighted BCE + Soft Dice Loss for crack localization heatmap.
    Weighted BCE overcomes 99.5% background sparsity to force sharp crack peaks.
    """
    pos_weight = torch.tensor([12.0], device=pred_heat.device)
    bce = F.binary_cross_entropy_with_logits(pred_heat, targets, pos_weight=pos_weight)
    
    pred_sig = torch.sigmoid(pred_heat)
    intersection = (pred_sig * targets).sum(dim=(2, 3))
    union = pred_sig.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-5
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
    
    return bce + 2.0 * dice_loss

def train_and_save_unified():
    device = torch.device('cpu')
    print(f"[BoneSense] Training Dual-Bone Calibrated Detection + Localization Model on {device}...", flush=True)
    
    anno_path = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared", "annotations.json")
    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        
    data_root = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared")
    train_ds = FracAtlasUnifiedDataset(data_root, annotations, split="train")
    val_ds = FracAtlasUnifiedDataset(data_root, annotations, split="val")
    test_ds = FracAtlasUnifiedDataset(data_root, annotations, split="test")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = DualBoneFractureNetwork(pretrained=False, num_classes=1, num_subtypes=5)
    
    ckpt_path = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_fracatlas_model.pt")
    if os.path.exists(ckpt_path):
        prev_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(prev_ckpt['model_state_dict'], strict=False)
        print(f"[BoneSense] Loaded existing backbone weights from {ckpt_path}", flush=True)
        
    model.to(device)
    
    # Train localization head + Spatial Attention + Relational Attention + Classifier
    optimizer = torch.optim.AdamW([
        {'params': model.loc_head.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        {'params': model.cbam.parameters(), 'lr': 2e-4},
        {'params': model.inter_bone_att.parameters(), 'lr': 2e-4},
        {'params': model.layer4.parameters(), 'lr': 5e-5},
        {'params': model.shared_fc.parameters(), 'lr': 1e-4},
        {'params': model.detection_head.parameters(), 'lr': 2e-4},
        {'params': model.subtype_head.parameters(), 'lr': 1e-4}
    ])
    
    epochs = 8
    best_val_loss = float('inf')
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            targets = batch["target_map"].to(device)
            weights = batch["weight"].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            det_logits, sub_logits, loc_heat = model(imgs, return_subtypes=True, return_localization=True)
            
            # Weighted classification loss
            raw_cls = F.binary_cross_entropy_with_logits(det_logits, labels, reduction='none')
            l_cls = (raw_cls * weights).mean()
            
            # Combined Weighted BCE + Dice Loss for localization
            l_loc = compute_combined_loc_loss(loc_heat, targets)
            
            loss = l_cls + 2.5 * l_loc
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                targets = batch["target_map"].to(device)
                weights = batch["weight"].to(device).unsqueeze(1)
                
                det_logits, _, loc_heat = model(imgs, return_subtypes=True, return_localization=True)
                raw_cls = F.binary_cross_entropy_with_logits(det_logits, labels, reduction='none')
                l_cls = (raw_cls * weights).mean()
                l_loc = compute_combined_loc_loss(loc_heat, targets)
                val_loss += (l_cls + 2.5 * l_loc).item() * imgs.size(0)
                
        val_loss /= len(val_ds)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "dataset": "FracAtlas_Unified_Calibrated"
            }
            torch.save(save_dict, ckpt_path)
            print(f"  -> Saved best unified model to {ckpt_path}", flush=True)
            
    print(f"\n[SUCCESS] Unified Model training completed in {(time.time() - t0)/60:.2f} minutes!", flush=True)
    
    # Evaluate on test set
    model.eval()
    errors = []
    frac_probs = []
    norm_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"]
            filenames = batch["filename"]
            
            det_logits, _, loc_heat = model(imgs, return_subtypes=True, return_localization=True)
            probs = torch.sigmoid(det_logits).squeeze().cpu().numpy()
            pred_heat = torch.sigmoid(loc_heat).cpu().numpy()
            
            for i in range(imgs.size(0)):
                p = float(probs[i]) if imgs.size(0) > 1 else float(probs)
                if labels[i] > 0.5:
                    frac_probs.append(p)
                    fn = filenames[i]
                    gt_boxes = annotations.get(fn, {}).get("bboxes_256", [])
                    if gt_boxes:
                        heat = pred_heat[i, 0]
                        py, px = np.unravel_index(np.argmax(heat), heat.shape)
                        pred_cx = px * 4.0
                        pred_cy = py * 4.0
                        
                        gt = gt_boxes[0]
                        gt_cx = (gt[0] + gt[2]) / 2.0
                        gt_cy = (gt[1] + gt[3]) / 2.0
                        dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
                        errors.append(dist)
                else:
                    norm_probs.append(p)
                    
    print(f"\n[TEST EVALUATION] Median localization distance: {np.median(errors):.1f} px (Mean: {np.mean(errors):.1f} px)", flush=True)
    print(f"[TEST EVALUATION] Within 25px: {(np.array(errors) <= 25).mean()*100:.1f}% | Within 40px: {(np.array(errors) <= 40).mean()*100:.1f}%", flush=True)
    print(f"[TEST EVALUATION] Fractures mean prob: {np.mean(frac_probs):.3f} | Normals mean prob: {np.mean(norm_probs):.3f}")
    
    # Check false alarm rate at thresholds 0.50 and 0.55
    for th in [0.50, 0.55, 0.58]:
        rec = (np.array(frac_probs) >= th).mean() * 100
        fp = (np.array(norm_probs) >= th).mean() * 100
        print(f"  Threshold {th:.2f} -> Fracture Recall: {rec:.1f}% | Normal FP Rate: {fp:.1f}%")

if __name__ == '__main__':
    train_and_save_unified()
