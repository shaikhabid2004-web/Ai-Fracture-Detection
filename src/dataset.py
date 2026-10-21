import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import csv
import torchvision.transforms as T

# Clinical Subtype Ground-Truth Dictionary based on Radiographic Assessment
# 0: Hairline / Fissure (Minor crack)
# 1: Transverse / Oblique (Linear break)
# 2: Comminuted (Multi-fragment)
# 3: Greenstick (Incomplete bending)
# 4: Dislocation / Severe Displacement
FRACTURE_SUBTYPE_MAP = {
    "img_001": 0, "img_014": 0,
    "img_004": 1, "img_008": 1, "img_013": 1, "img_018": 1, "img_020": 1, "img_023": 1, "img_027": 1, "img_029": 1,
    "img_005": 2, "img_007": 2, "img_009": 2, "img_011": 2, "img_015": 2, "img_017": 2, "img_025": 2, "img_026": 2,
    "img_010": 3, "img_016": 3, "img_019": 3, "img_022": 3,
    "img_002": 4, "img_006": 4, "img_012": 4, "img_024": 4, "img_028": 4
}

class BoneFractureDataset(Dataset):
    """
    PyTorch Dataset for cleaned Bone Fracture X-rays.
    Supports dual-view (AP & Lateral orthogonal projections) and clinical subtype labels.
    """
    def __init__(self, data_root=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data", 
                 mode="unified_512", is_train=True, transform=None):
        self.data_root = data_root
        self.mode = mode
        self.is_train = is_train
        self.split_views_dir = os.path.join(data_root, "cleaned", "split_views")
        
        self.samples = []
        if mode in ("train", "test"):
            split_dir = os.path.join(data_root, "dataset_splits", mode)
            frac_dir = os.path.join(split_dir, "fracture")
            norm_dir = os.path.join(split_dir, "normal")
            
            if os.path.exists(frac_dir):
                for f in sorted(os.listdir(frac_dir)):
                    if f.endswith('.png') or f.endswith('.jpg'):
                        self.samples.append((os.path.join(frac_dir, f), 1.0, f))
            if os.path.exists(norm_dir):
                for f in sorted(os.listdir(norm_dir)):
                    if f.endswith('.png') or f.endswith('.jpg'):
                        self.samples.append((os.path.join(norm_dir, f), 0.0, f))
                        
            self.image_files = [s[2] for s in self.samples]
        elif mode == "unified_512":
            self.img_dir = os.path.join(data_root, "cleaned", "unified_512")
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
            self.samples = [(os.path.join(self.img_dir, f), 1.0, f) for f in self.image_files]
        elif mode == "split_views":
            self.img_dir = os.path.join(data_root, "cleaned", "split_views")
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
            self.samples = [(os.path.join(self.img_dir, f), 1.0, f) for f in self.image_files]
        elif mode == "unified_1024":
            self.img_dir = os.path.join(data_root, "cleaned", "unified_1024")
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
            self.samples = [(os.path.join(self.img_dir, f), 1.0, f) for f in self.image_files]
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        manifest_path = os.path.join(data_root, "dataset_manifest.csv")
        self.manifest = []
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = list(csv.DictReader(f))
        
        if transform is not None:
            self.transform = transform
        else:
            if is_train:
                self.transform = T.Compose([
                    T.Resize((512, 512)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=7),
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5])
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5])
                ])
                
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        fpath, lbl_val, fname = self.samples[idx]
        base_id = os.path.splitext(fname)[0].replace("_AP", "").replace("_LAT", "")
        
        # Determine subtype label
        if lbl_val == 1.0:
            subtype_val = FRACTURE_SUBTYPE_MAP.get(base_id, 1) # Default to Transverse/Oblique
        else:
            subtype_val = 0 # Normal (not used in subtype loss)
            
        img = Image.open(fpath).convert('L') # Grayscale radiograph
        
        # Check if individual AP and LAT split views exist
        ap_path = os.path.join(self.split_views_dir, f"{base_id}_AP.png")
        lat_path = os.path.join(self.split_views_dir, f"{base_id}_LAT.png")
        
        if os.path.exists(ap_path) and os.path.exists(lat_path):
            img_ap = Image.open(ap_path).convert('L')
            img_lat = Image.open(lat_path).convert('L')
        else:
            # If split views do not exist, use identical projections
            img_ap = img
            img_lat = img
            
        tensor = self.transform(img)
        tensor_ap = self.transform(img_ap)
        tensor_lat = self.transform(img_lat)
        
        label = torch.tensor(lbl_val, dtype=torch.float32)
        subtype = torch.tensor(subtype_val, dtype=torch.long)
        
        return {
            "image": tensor,
            "image_ap": tensor_ap,
            "image_lat": tensor_lat,
            "label": label,
            "subtype": subtype,
            "filename": fname
        }

if __name__ == "__main__":
    ds = BoneFractureDataset(mode="train")
    print(f"Loaded {len(ds)} samples in train mode.")
    sample = ds[0]
    print(f"Sample tensor shapes -> AP: {sample['image_ap'].shape}, LAT: {sample['image_lat'].shape}, Det Label: {sample['label']}, Subtype: {sample['subtype']}")

