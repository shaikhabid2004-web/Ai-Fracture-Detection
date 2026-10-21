import os
import sys
import csv
import time
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

ANATOMY_CLASSES = ["Shoulder", "Leg", "Hand", "Hip"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "anatomy_classifier.pt")

class AnatomyClassifier(nn.Module):
    def __init__(self, in_features=128, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

def get_anatomical_zone_description(body_part, y_norm, x_norm, is_fracture=True):
    """
    Returns an anatomically accurate orthopedic description of the region.
    """
    if not is_fracture:
        if body_part == "Shoulder":
            return "Shoulder Girdle [Intact Clavicle, Scapula & Proximal Humerus]"
        elif body_part == "Hip":
            return "Pelvic Ring & Hip [Intact Femoral Head & Acetabulum]"
        elif body_part == "Hand":
            return "Hand / Wrist [Intact Metacarpals & Phalangeal Rays]"
        elif body_part == "Leg":
            return "Lower Extremity [Intact Tibia-Fibula Cortical Alignment]"
        return "Normal / Intact Skeletal Structure"
        
    if body_part == "Shoulder":
        if y_norm < 0.38:
            zone = "Clavicle & Acromioclavicular (AC) Joint"
        elif x_norm > 0.45 and y_norm >= 0.38:
            zone = "Proximal Humerus / Greater Tuberosity & Surgical Neck"
        else:
            zone = "Scapulothoracic Complex / Glenoid Neck"
        return f"Shoulder Girdle [{zone}]"
        
    elif body_part == "Hip":
        if y_norm < 0.50:
            zone = "Iliac Crest & Acetabular Roof"
        else:
            zone = "Femoral Neck & Intertrochanteric Region"
        return f"Pelvis & Hip [{zone}]"
        
    elif body_part == "Hand":
        if y_norm < 0.35:
            zone = "Phalangeal Ray (Distal / Proximal Phalanx)"
        elif y_norm < 0.70:
            zone = "Metacarpal Shaft & Head"
        else:
            zone = "Carpal Arch & Distal Radius/Ulna Joint"
        return f"Hand / Wrist [{zone}]"
        
    else: # Leg / Ankle
        if y_norm < 0.35:
            zone = "Proximal Tibia / Fibular Head"
        elif y_norm <= 0.65:
            zone = "Diaphyseal Shaft [Cortical Alignment]"
        else:
            zone = "Distal / Metaphyseal Ankle Mortise"
        return f"Lower Extremity [{zone}]"

def train_and_save_anatomy_classifier(fracatlas_root, model_backbone, device='cpu'):
    """
    Extracts features and trains a high-precision anatomy classifier.
    """
    csv_file = os.path.join(fracatlas_root, "dataset.csv")
    img_dir = os.path.join(fracatlas_root, "images")
    
    body_part_keys = ["shoulder", "leg", "hand", "hip"]
    samples = []
    
    with open(csv_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["image_id"]
            p_f = os.path.join(img_dir, "Fractured", fn)
            p_n = os.path.join(img_dir, "Non_fractured", fn)
            fpath = p_f if os.path.exists(p_f) else (p_n if os.path.exists(p_n) else None)
            if not fpath: continue
            
            for idx, key in enumerate(body_part_keys):
                if row.get(key) == "1":
                    samples.append((fpath, idx, fn))
                    break
                    
    import random
    random.seed(42)
    by_class = {i: [] for i in range(4)}
    for s in samples:
        by_class[s[1]].append(s)
        
    # Balanced training on up to 350 per class (includes all shoulders)
    balanced = []
    for c in range(4):
        random.shuffle(by_class[c])
        balanced.extend(by_class[c][:350])
        
    random.shuffle(balanced)
    print(f"[Anatomy Engine] Training on {len(balanced)} balanced medical radiographs at native 256x256...")
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    feats = []
    labels = []
    model_backbone.eval()
    
    for fpath, lbl, fn in balanced:
        try:
            with Image.open(fpath) as img:
                t = transform(img.convert('L')).unsqueeze(0).to(device)
                with torch.no_grad():
                    f3, f4 = model_backbone.extract_features(t)
                    pool = model_backbone.avgpool(f4).flatten(1)
                    shared = model_backbone.shared_fc(pool)
                    feats.append(shared.cpu().squeeze(0))
                    labels.append(lbl)
        except Exception:
            continue
            
    X = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    
    classifier = AnatomyClassifier(in_features=128, num_classes=4).to(device)
    classifier.train()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(120):
        optimizer.zero_grad()
        out = classifier(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
    classifier.eval()
    with torch.no_grad():
        preds = classifier(X).argmax(dim=-1)
        train_acc = (preds == y).float().mean().item() * 100.0
        print(f"[Anatomy Engine] Training complete! Accuracy: {train_acc:.1f}%")
        
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(classifier.state_dict(), CHECKPOINT_PATH)
    print(f"[Anatomy Engine] Saved model to: {CHECKPOINT_PATH}")
    return classifier

def load_anatomy_classifier(device='cpu'):
    classifier = AnatomyClassifier(in_features=128, num_classes=4).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        classifier.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        classifier.eval()
        return classifier
    return None
