import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork

class FracAtlasDataset(Dataset):
    """
    High-Speed PyTorch Dataset for FracAtlas Balanced Benchmark.
    Reads pre-cached 256x256 radiographs.
    """
    def __init__(self, data_root, split="train", is_train=True):
        self.split_dir = os.path.join(data_root, split)
        self.samples = []
        
        frac_dir = os.path.join(self.split_dir, "fracture")
        norm_dir = os.path.join(self.split_dir, "normal")
        
        if os.path.exists(frac_dir):
            for f in sorted(os.listdir(frac_dir)):
                if f.endswith('.png'):
                    self.samples.append((os.path.join(frac_dir, f), 1.0, f))
        if os.path.exists(norm_dir):
            for f in sorted(os.listdir(norm_dir)):
                if f.endswith('.png'):
                    self.samples.append((os.path.join(norm_dir, f), 0.0, f))
                    
        if is_train:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        fpath, label, fname = self.samples[idx]
        img = Image.open(fpath).convert("L")
        tensor = self.transform(img)
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "filename": fname
        }

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            
            det_logits, _ = model(imgs, return_subtypes=True)
            loss = criterion(det_logits, labels)
            total_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(det_logits).cpu().numpy().flatten()
            preds = (probs >= 0.50).astype(int)
            targets = labels.cpu().numpy().flatten().astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    n = len(all_targets)
    avg_loss = total_loss / n if n > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec = precision_score(all_targets, all_preds, zero_division=0) * 100.0
    rec = recall_score(all_targets, all_preds, zero_division=0) * 100.0
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 1.0 if acc == 100.0 else 0.5
        
    return avg_loss, acc, prec, rec, f1, auc

def train_fracatlas(
    data_root=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\fracatlas_prepared",
    checkpoint_dir=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\checkpoints",
    epochs=6,
    batch_size=16,
    learning_rate=3e-4
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"      FAST FRACATLAS TRAINING PIPELINE ON {device}")
    print("=" * 80)
    
    train_ds = FracAtlasDataset(data_root, split="train", is_train=True)
    val_ds = FracAtlasDataset(data_root, split="val", is_train=False)
    test_ds = FracAtlasDataset(data_root, split="test", is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print(f"Loaded {len(train_ds)} Training samples ({len(train_loader)} batches/epoch)")
    print(f"Loaded {len(val_ds)} Validation samples")
    print(f"Loaded {len(test_ds)} Independent Test samples")
    
    # Initialize DualBoneFractureNetwork with ResNet backbone and spatial attention
    model = DualBoneFractureNetwork(pretrained=True, num_classes=1, num_subtypes=5).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = -1.0
    best_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "best_fracatlas_model.pt")
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            det_logits, _ = model(imgs, return_subtypes=True)
            loss = criterion(det_logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            
        scheduler.step()
        train_loss /= len(train_ds)
        
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_model(model, val_loader, device, criterion)
        epoch_sec = time.time() - epoch_start
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}% | "
              f"F1: {val_f1:.4f} | "
              f"AUC: {val_auc:.4f}")
              
        if val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss):
            best_f1 = val_f1
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "dataset": "FracAtlas"
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path} (Val F1: {val_f1:.4f})")
            
    total_sec = time.time() - start_time
    print(f"\nTraining completed in {total_sec / 60.0:.2f} minutes!")
    
    # Final Evaluation on Independent Test Set
    print("\nEvaluating best model on unseen Test Set...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate_model(model, test_loader, device, criterion)
    
    print("=" * 80)
    print("      FINAL INDEPENDENT TEST PERFORMANCE (FRACATLAS)")
    print("=" * 80)
    print(f"Test Accuracy   : {test_acc:.2f}%")
    print(f"Test Precision  : {test_prec:.2f}%")
    print(f"Test Recall     : {test_rec:.2f}%")
    print(f"Test F1-Score   : {test_f1:.4f}")
    print(f"Test ROC-AUC    : {test_auc:.4f}")
    print("=" * 80)
    
    metrics = {
        "dataset": "FracAtlas",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "test_roc_auc": test_auc,
        "training_time_min": round(total_sec / 60.0, 2),
        "best_epoch": checkpoint["epoch"],
        "checkpoint": ckpt_path
    }
    
    metrics_path = os.path.join(data_root, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
        
    print(f"Test metrics saved to: {metrics_path}")
    return metrics

if __name__ == "__main__":
    train_fracatlas(epochs=6, batch_size=16)
