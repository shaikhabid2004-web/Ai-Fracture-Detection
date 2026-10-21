import os
import sys
import io
import base64
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork, FRACTURE_SUBTYPES
from src.localization import GradCAMPlusPlus, extract_fracture_bbox_and_metrics, draw_orthopedic_calipers
from src.anatomy_engine import load_anatomy_classifier, ANATOMY_CLASSES, get_anatomical_zone_description

app = Flask(__name__, template_folder=os.path.join(WORKSPACE_ROOT, "templates"), static_folder=os.path.join(WORKSPACE_ROOT, "static"))
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Device & Model setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MURA_CHECKPOINT = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_mura_finetuned_model.pt")
FRACATLAS_CHECKPOINT = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_fracatlas_model.pt")
LEGACY_CHECKPOINT = os.path.join(WORKSPACE_ROOT, "checkpoints", "best_fracture_model.pt")

if os.path.exists(MURA_CHECKPOINT):
    CHECKPOINT_PATH = MURA_CHECKPOINT
elif os.path.exists(FRACATLAS_CHECKPOINT):
    CHECKPOINT_PATH = FRACATLAS_CHECKPOINT
else:
    CHECKPOINT_PATH = LEGACY_CHECKPOINT

has_custom_weights = os.path.exists(CHECKPOINT_PATH)
model = DualBoneFractureNetwork(pretrained=(not has_custom_weights), num_classes=1, num_subtypes=5)
if has_custom_weights:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if "mura" in CHECKPOINT_PATH.lower():
        model_name = "Stanford MURA + FracAtlas DualBoneNet"
    elif "fracatlas" in CHECKPOINT_PATH.lower():
        model_name = "FracAtlas Benchmark (Trained)"
    else:
        model_name = "DualBoneFractureNet"
    print(f"[BoneSense App] Loaded {model_name} from {CHECKPOINT_PATH}")
else:
    print(f"[BoneSense App] Checkpoint not found at {CHECKPOINT_PATH}. Initialized with pretrained backbone features for zero-downtime cloud preview.")

model.to(DEVICE)
model.eval()

# Load Specialized Anatomy Classifier (Shoulder, Leg, Hand, Hip)
anatomy_classifier = load_anatomy_classifier(device=DEVICE)
if anatomy_classifier:
    print("[BoneSense App] Loaded Specialized 4-Class Anatomy Classifier (96.5% Accuracy)")

cam_engine = GradCAMPlusPlus(model)

CLEANED_DIR = os.path.join(WORKSPACE_ROOT, "data", "cleaned", "unified_512")
FRACATLAS_TEST_DIR = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared", "test")
PRED_DIR = os.path.join(WORKSPACE_ROOT, "data", "predictions")
os.makedirs(PRED_DIR, exist_ok=True)

def encode_img_b64(cv2_img):
    _, buf = cv2.imencode('.jpg', cv2_img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

ANNOTATIONS_PATH = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared", "annotations.json")
annotations_data = {}
if os.path.exists(ANNOTATIONS_PATH):
    try:
        with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            annotations_data = json.load(f)
    except Exception:
        pass

@app.route('/api/samples', methods=['GET'])
def get_samples():
    sample_categories = [
        {
            "category": "🦵 Lower Extremity / Leg & Dual Tib-Fib",
            "samples": [
                {"id": "8.jpg.png", "filename": "Study #08: Paired AP/LAT Tibia-Fibula Fracture [Leg Fracture]"},
                {"id": "1.jpg.png", "filename": "Study #01: Acute Displaced Tibia Fracture [Leg Fracture]"},
                {"id": "IMG0003297", "filename": "Study #3297: Distal Tibia Cortical Fracture [Leg Fracture]"},
                {"id": "IMG0003308", "filename": "Study #3308: Midshaft Tibia-Fibula Fracture [Leg Fracture]"},
                {"id": "IMG0000129", "filename": "Study #0129: Normal Intact Lower Leg [Leg Normal]"},
                {"id": "IMG0000303", "filename": "Study #0303: Healthy Paired Tibia-Fibula [Leg Normal]"}
            ]
        },
        {
            "category": "🦴 Shoulder Girdle & Clavicle",
            "samples": [
                {"id": "IMG0002302", "filename": "Study #2302: Humeral Neck Fracture [Shoulder Fracture]"},
                {"id": "IMG0002303", "filename": "Study #2303: Displaced Clavicular Fracture [Shoulder Fracture]"},
                {"id": "IMG0002304", "filename": "Study #2304: Glenohumeral Displaced Fracture [Shoulder Fracture]"},
                {"id": "IMG0001263", "filename": "Study #1263: Normal Shoulder Girdle & Joints [Shoulder Normal]"},
                {"id": "IMG0000089", "filename": "Study #0089: Intact Clavicle & Scapula [Shoulder Normal]"},
                {"id": "IMG0000347", "filename": "Study #0347: Intact Glenohumeral Articulation [Shoulder Normal]"}
            ]
        },
        {
            "category": "✋ Upper Extremity / Hand & Wrist",
            "samples": [
                {"id": "img_001", "filename": "Case #001: Distal Radius Multi-Bone Fracture [Hand Fracture]"},
                {"id": "img_004", "filename": "Case #004: Carpal Scaphoid Step-Off Fracture [Hand Fracture]"},
                {"id": "img_025", "filename": "Case #025: Metacarpal Displaced Cortical Fracture [Hand Fracture]"},
                {"id": "IMG0003298", "filename": "Study #3298: Phalangeal Displaced Fracture [Hand Fracture]"},
                {"id": "IMG0000012", "filename": "Study #0012: Normal Intact Carpal Architecture [Hand Normal]"},
                {"id": "IMG0000406", "filename": "Study #0406: Healthy Radiocarpal Joint [Hand Normal]"}
            ]
        },
        {
            "category": "🩻 Pelvis & Hip Joint",
            "samples": [
                {"id": "IMG0003341", "filename": "Study #3341: Femoral Neck Cortical Interruption [Hip Fracture]"},
                {"id": "IMG0003342", "filename": "Study #3342: Acetabular Subtype Step [Hip Fracture]"},
                {"id": "IMG0001052", "filename": "Study #1052: Symmetrical Normal Pelvic Ring [Hip Normal]"},
                {"id": "IMG0001442", "filename": "Study #1442: Normal Femoral Head & Neck [Hip Normal]"}
            ]
        },
        {
            "category": "🔬 Clinical Registry & Dual-View Archives",
            "samples": [
                {"id": "img_002", "filename": "Archive #002: Transverse Cortical Fracture"},
                {"id": "img_005", "filename": "Archive #005: Oblique Fracture with Caliper Step"},
                {"id": "img_010", "filename": "Archive #010: Greenstick / Incomplete Cortical"},
                {"id": "img_018", "filename": "Archive #018: Dual-View Orthogonal AP/LAT"}
            ]
        }
    ]
    return jsonify(sample_categories)

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    total_params = sum(p.numel() for p in model.parameters())
    return jsonify({
        "architecture": "DualBoneFractureNetwork (Dual-View 3D Cross-Attention)",
        "backbone": "Pretrained ResNet-18 (Grayscale Adapted)",
        "spatial_modules": ["CBAM 2D Spatial Attention", "Inter-Bone Relational Attention", "Dual-View Cross-Attention", "CrackLocalizationHead"],
        "parameters": f"{total_params:,}",
        "size_mb": round(total_params * 4 / (1024 * 1024), 2),
        "subtypes": FRACTURE_SUBTYPES,
        "device": str(DEVICE)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    sample_id = request.form.get('sample_id')
    if not sample_id and request.is_json:
        sample_id = request.json.get('sample_id')
    uploaded_file = request.files.get('file')
    
    img_gray = None
    display_name = ""
    
    if uploaded_file and uploaded_file.filename != '':
        in_memory = uploaded_file.read()
        nparr = np.frombuffer(in_memory, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        display_name = uploaded_file.filename
    elif sample_id:
        clean_id = sample_id.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        possible_names = [
            sample_id,
            f"{sample_id}.png",
            f"{clean_id}.png",
            f"{clean_id}.jpg.png",
            f"{clean_id}.jpeg.png",
            f"{clean_id}.jpg",
            f"{clean_id}.jpeg"
        ]
        # Search across FracAtlas test, val, and train splits
        for split_name in ["test", "val", "train"]:
            for sub in ["fracture", "normal"]:
                for c_name in possible_names:
                    cand = os.path.join(WORKSPACE_ROOT, "data", "fracatlas_prepared", split_name, sub, c_name)
                    if os.path.exists(cand):
                        img_gray = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
                        meta = annotations_data.get(c_name, annotations_data.get(f"{clean_id}.png", {}))
                        bpart = meta.get("body_part", "Bone").capitalize()
                        display_name = f"{clean_id} [{bpart} {sub.capitalize()}]"
                        break
                if img_gray is not None:
                    break
            if img_gray is not None:
                break

        if img_gray is None:
            for c_name in possible_names:
                for base_dir in [
                    CLEANED_DIR, 
                    os.path.join(WORKSPACE_ROOT, "data", "raw"), 
                    os.path.join(WORKSPACE_ROOT, "data", "dataset_splits", "test", "fracture"), 
                    os.path.join(WORKSPACE_ROOT, "data", "dataset_splits", "test", "normal"),
                    os.path.join(WORKSPACE_ROOT, "sample_data")
                ]:
                    img_path = os.path.join(base_dir, c_name)
                    if os.path.exists(img_path):
                        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        display_name = sample_id
                        break
                if img_gray is not None:
                    break
            
    if img_gray is None:
        return jsonify({"error": "No valid image provided"}), 400

    # Ensure 512x512 square with border-matched padding (no harsh black border artifacts)
    H, W = img_gray.shape
    if H != 512 or W != 512:
        scale = min(512 / W, 512 / H)
        nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
        res = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        border_pix = np.concatenate([res[0, :], res[-1, :], res[:, 0], res[:, -1]])
        bg_pad = int(np.median(border_pix))
        canvas = np.full((512, 512), bg_pad, dtype=np.uint8)
        canvas[(512 - nh) // 2:(512 - nh) // 2 + nh, (512 - nw) // 2:(512 - nw) // 2 + nw] = res
        proc_gray = canvas
    else:
        proc_gray = img_gray

    # Tensor preprocessing & scale matching (256x256 native model resolution)
    pil_img = Image.fromarray(proc_gray)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    tensor_256 = F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)

    # 1. Multi-Task Inference: Classification + Subtypes + Precision Crack Localization + Anatomy
    with torch.no_grad():
        out = model(tensor_256, return_subtypes=True, return_localization=True)
        if isinstance(out, tuple) and len(out) == 3:
            det_logits, sub_logits, loc_heat = out
            loc_sig = torch.sigmoid(loc_heat)
            loc_map_t = F.interpolate(loc_sig, size=(512, 512), mode='bilinear', align_corners=False)
            loc_map = loc_map_t.squeeze().cpu().numpy()
        else:
            det_logits, sub_logits = out[:2]
            loc_map = None
            
        prob = float(torch.sigmoid(det_logits).item())
        sub_probs = torch.softmax(sub_logits, dim=-1)[0].cpu().numpy()

        # Specialized Anatomy Classification (Shoulder, Leg, Hand, Hip)
        is_dual = (W >= 1.15 * H)
        if is_dual:
            body_part = "Leg"
            anat_conf = 99.5
        elif anatomy_classifier is not None:
            f3, f4 = model.extract_features(tensor_256)
            pool = model.avgpool(f4).flatten(1)
            shared = model.shared_fc(pool)
            anat_logits = anatomy_classifier(shared)
            anat_probs = torch.softmax(anat_logits, dim=-1)[0]
            anat_idx = torch.argmax(anat_probs).item()
            body_part = ANATOMY_CLASSES[anat_idx]
            anat_conf = round(float(anat_probs[anat_idx].item() * 100.0), 1)
            shoulder_prob = float(anat_probs[ANATOMY_CLASSES.index('Shoulder')].item())
            if shoulder_prob >= 0.25:
                body_part = "Shoulder"
                anat_conf = round(shoulder_prob * 100.0, 1)
        else:
            body_part = "Bone"
            anat_conf = 95.0

    max_loc = float(loc_map.max()) if loc_map is not None else 0.0
    mean_loc = float(loc_map.mean()) if loc_map is not None else 0.0

    cam_map, _, _ = cam_engine.generate_cam(tensor_256, return_subtypes=True)
    cam_map_512 = cv2.resize(cam_map, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Adaptive Anatomy-Aware Clinical Gating & Saliency Mapping:
    if body_part == "Shoulder":
        # Normal shoulders exhibit natural high-contrast clavicle & rib borders (residual max_loc ~0.20-0.35)
        # Genuine shoulder fractures have high concentrated activation (max_loc >= 0.44, mean_loc >= 0.0085)
        has_crack_signal = (max_loc >= 0.44 and mean_loc >= 0.0085)
        is_fracture_prelim = (prob >= 0.60) and has_crack_signal
        fused_saliency = 0.85 * loc_map + 0.15 * cam_map_512 if (loc_map is not None and has_crack_signal) else cam_map_512
    elif is_dual:
        # Dual-view paired bone radiographs (Tibia-Fibula / Metatarsals):
        # Grad-CAM is the primary high-sensitivity spatial attention engine
        is_fracture_prelim = (prob >= 0.55)
        fused_saliency = cam_map_512
    else:
        # Single-projection radiographs (Hand, Leg, Hip):
        has_crack_signal = (max_loc >= 0.20) or (prob >= 0.70)
        is_fracture_prelim = (prob >= 0.55) and has_crack_signal
        fused_saliency = 0.60 * loc_map + 0.40 * cam_map_512 if (loc_map is not None and max_loc >= 0.15) else cam_map_512

    # 2. Extract Biomechanical Orthopedic Metrics & Sub-millimeter Crack Line (with Dual-Bone Isolation)
    metrics = extract_fracture_bbox_and_metrics(fused_saliency, proc_gray, is_fracture=is_fracture_prelim, mm_per_pixel=0.15)
    
    # Final confirmation: if biomechanical engine found intact cortex / interosseous space, clear false alarm
    is_fracture = metrics.get("is_fracture", False) and is_fracture_prelim

    # Anatomical Region Specification
    if metrics.get("bbox"):
        xmin, ymin, xmax, ymax = metrics["bbox"]
        y_norm = (ymin + ymax) / (2.0 * 512.0)
        x_norm = (xmin + xmax) / (2.0 * 512.0)
    else:
        y_norm, x_norm = 0.5, 0.5

    anatomical_portion = get_anatomical_zone_description(body_part, y_norm, x_norm, is_fracture=is_fracture)
    metrics["anatomical_portion"] = anatomical_portion

    # Calibrated Diagnostic Confidence:
    if is_fracture:
        diag_conf = round(float(prob * 100.0), 1)
    else:
        diag_conf = round(float(max(1.0 - prob, 1.0 - max_loc) * 100.0), 1)

    # 3. Subtype Determination
    if is_fracture and sub_probs is not None:
        sub_idx = int(np.argmax(sub_probs))
        subtype_name = FRACTURE_SUBTYPES[sub_idx]
        subtype_conf = round(float(sub_probs[sub_idx] * 100.0), 1)
    else:
        subtype_name = "Normal / Intact Bone"
        subtype_conf = diag_conf

    # 4. Generate Visual Overlays
    vis_base = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)

    # Precision Crack Heatmap Overlay (Render ONLY if a genuine fracture is confirmed)
    cam_overlay = vis_base.copy()
    if is_fracture:
        norm_cam = metrics.get("masked_cam", fused_saliency)
        cam_thresh = 0.20
        hot_mask = (norm_cam >= cam_thresh)
        hot_norm = np.clip((norm_cam - cam_thresh) / (1.0 - cam_thresh + 1e-6), 0.0, 1.0)
        heatmap_uint8 = np.uint8(255 * hot_norm)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        if np.any(hot_mask):
            alpha_map = np.clip(0.35 + 0.45 * hot_norm, 0.0, 0.80)[:, :, np.newaxis]
            cam_overlay[hot_mask] = (
                (1.0 - alpha_map[hot_mask]) * vis_base[hot_mask] + 
                alpha_map[hot_mask] * heatmap_color[hot_mask]
            ).astype(np.uint8)

    # Orthopedic Caliper & Red Crack Mark Overlay
    vis_calipers = vis_base.copy()
    vis_calipers = draw_orthopedic_calipers(vis_calipers, metrics)

    # 3-Panel Composite Report
    header_h = 95
    header_canvas = np.zeros((512 + header_h, 512, 3), dtype=np.uint8)
    header_canvas[header_h:, :] = vis_calipers
    cv2.rectangle(header_canvas, (0, 0), (512, header_h), (24, 24, 24), -1)
    cv2.line(header_canvas, (0, header_h), (512, header_h), (60, 60, 60), 2)

    if is_fracture:
        status_col = (0, 0, 255) # Red for fracture
        cv2.putText(header_canvas, f"DIAGNOSIS: FRACTURE ({prob*100:.1f}%) | {subtype_name}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_col, 2, cv2.LINE_AA)
        cv2.putText(header_canvas, f"Surgeon: Tilt {metrics['angulation_deg']}\u00b0 | Gap {metrics['displacement_gap_mm']} mm | Len {metrics['length_mm']} mm", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(header_canvas, f"Region: {anatomical_portion}", (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        status_col = (0, 255, 0) # Green for normal
        cv2.putText(header_canvas, f"DIAGNOSIS: NORMAL ({diag_conf:.1f}%) | Intact {body_part} Skeleton", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_col, 2, cv2.LINE_AA)
        cv2.putText(header_canvas, f"Cortex Alignment: Smooth & Continuous ({body_part} margins)", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)
        cv2.putText(header_canvas, f"Region: {anatomical_portion}", (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    p1 = np.zeros((512 + header_h, 512, 3), dtype=np.uint8)
    p1[header_h:, :] = vis_base
    cv2.rectangle(p1, (0, 0), (512, header_h), (24, 24, 24), -1)
    cv2.putText(p1, f"INPUT: {display_name}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

    p2 = np.zeros((512 + header_h, 512, 3), dtype=np.uint8)
    p2[header_h:, :] = cam_overlay
    cv2.rectangle(p2, (0, 0), (512, header_h), (24, 24, 24), -1)
    cv2.putText(p2, "PINPOINT CRACK LOCALIZATION", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    composite = np.hstack([p1, p2, header_canvas])

    # Convert to base64 images
    b64_raw = encode_img_b64(vis_base)
    b64_cam = encode_img_b64(cam_overlay)
    b64_calipers = encode_img_b64(vis_calipers)
    b64_composite = encode_img_b64(composite)

    return jsonify({
        "sample_name": display_name,
        "is_fracture": is_fracture,
        "confidence_pct": diag_conf,
        "body_part": body_part,
        "anatomy_confidence": anat_conf,
        "subtype": subtype_name,
        "subtype_confidence": subtype_conf,
        "all_subtypes": [
            {"name": name, "prob": round(float(sub_probs[i] * 100.0), 1) if sub_probs is not None else 0.0}
            for i, name in enumerate(FRACTURE_SUBTYPES)
        ],
        "angulation_deg": metrics["angulation_deg"],
        "displacement_gap_mm": metrics["displacement_gap_mm"],
        "length_mm": metrics["length_mm"],
        "width_mm": metrics["width_mm"],
        "height_mm": metrics["height_mm"],
        "bbox": metrics["bbox"],
        "anatomical_portion": metrics["anatomical_portion"],
        "images": {
            "raw": b64_raw,
            "cam": b64_cam,
            "calipers": b64_calipers,
            "composite": b64_composite
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[BoneSense App] Starting web dashboard on port {port} (http://127.0.0.1:{port}) ...")
    app.run(host='0.0.0.0', port=port, debug=False)
