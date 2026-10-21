import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DualBoneFractureNetwork, FRACTURE_SUBTYPES
from src.localization import GradCAMPlusPlus, extract_fracture_bbox_and_metrics, draw_orthopedic_calipers

def load_trained_model(checkpoint_path=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\checkpoints\best_fracture_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBoneFractureNetwork(pretrained=False, num_classes=1, num_subtypes=5)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained model weights from: {checkpoint_path} (Trained Epoch {checkpoint.get('epoch', '?')})")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running with initial weights.")
        
    model.to(device)
    model.eval()
    return model, device

def run_diagnostic_inference(image_path, model, device, output_dir=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\predictions", mm_per_pixel=0.15):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Load grayscale image
    img_pil = Image.open(image_path).convert('L')
    orig_np = np.array(img_pil)
    H, W = orig_np.shape
    
    # Preprocess tensor
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device) # 1 x 1 x H x W
    
    # Run Grad-CAM++ with Subtype Extraction
    cam_engine = GradCAMPlusPlus(model)
    cam_map, prob, sub_probs = cam_engine.generate_cam(tensor, return_subtypes=True)
    
    # Classify Subtype
    if prob >= 0.50 and sub_probs is not None:
        sub_idx = int(np.argmax(sub_probs))
        subtype_name = FRACTURE_SUBTYPES[sub_idx]
        subtype_conf = sub_probs[sub_idx] * 100.0
        status_color = (0, 255, 0) # Green for fracture
    else:
        subtype_name = "Normal / Intact Bone"
        subtype_conf = (1.0 - prob) * 100.0
        status_color = (255, 200, 0) # Amber/Cyan for normal
    
    # Extract Bbox and Orthopedic Biomechanical Metrics
    metrics = extract_fracture_bbox_and_metrics(cam_map, orig_np, threshold_ratio=0.45, mm_per_pixel=mm_per_pixel)
    
    # Visualization Construction
    vis_base = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2BGR)
    
    # Crisp Medical Heatmap Overlay
    norm_cam = metrics.get("masked_cam", cam_map)
    cam_thresh = 0.25
    hot_mask = (norm_cam >= cam_thresh)
    
    hot_norm = np.clip((norm_cam - cam_thresh) / (1.0 - cam_thresh + 1e-6), 0.0, 1.0)
    heatmap_uint8 = np.uint8(255 * hot_norm)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    cam_overlay = vis_base.copy()
    if np.any(hot_mask):
        alpha_map = np.clip(0.35 + 0.40 * hot_norm, 0.0, 0.75)[:, :, np.newaxis]
        cam_overlay[hot_mask] = (
            (1.0 - alpha_map[hot_mask]) * vis_base[hot_mask] + 
            alpha_map[hot_mask] * heatmap_color[hot_mask]
        ).astype(np.uint8)
    
    # Detection with Caliper Annotations
    vis_annotated = vis_base.copy()
    vis_annotated = draw_orthopedic_calipers(vis_annotated, metrics)
    
    # Diagnostic Information Banner
    header_h = 100
    header_canvas = np.zeros((H + header_h, W, 3), dtype=np.uint8)
    header_canvas[header_h:, :] = vis_annotated
    
    card_bg = (24, 24, 24)
    cv2.rectangle(header_canvas, (0, 0), (W, header_h), card_bg, -1)
    cv2.line(header_canvas, (0, header_h), (W, header_h), (60, 60, 60), 2)
    
    # Text annotations
    line1 = f"FRACTURE: {prob * 100:.1f}% | Subtype: {subtype_name} ({subtype_conf:.0f}%)"
    line2 = f"Surgeon Calipers: Tilt {metrics['angulation_deg']}\u00b0 | Gap {metrics['displacement_gap_mm']} mm | Len {metrics['length_mm']} mm"
    line3 = f"Anatomy: {metrics['anatomical_portion']} | Bbox: {metrics['bbox']}"
    
    cv2.putText(header_canvas, line1, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2, cv2.LINE_AA)
    cv2.putText(header_canvas, line2, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(header_canvas, line3, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Composite panel construction
    vis_base_padded = np.zeros((H + header_h, W, 3), dtype=np.uint8)
    vis_base_padded[header_h:, :] = vis_base
    cv2.rectangle(vis_base_padded, (0, 0), (W, header_h), card_bg, -1)
    cv2.putText(vis_base_padded, f"INPUT RADIOGRAPH: {base_name}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (220, 220, 220), 2)

    cam_overlay_padded = np.zeros((H + header_h, W, 3), dtype=np.uint8)
    cam_overlay_padded[header_h:, :] = cam_overlay
    cv2.rectangle(cam_overlay_padded, (0, 0), (W, header_h), card_bg, -1)
    cv2.putText(cam_overlay_padded, "DUAL-VIEW 3D CROSS-ATTN CAM", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 220, 255), 2)
    
    # Composite: [Raw Input] + [Dual-View CAM Heatmap] + [Orthopedic Calipers & Subtype Report]
    composite = np.hstack([vis_base_padded, cam_overlay_padded, header_canvas])
    
    # Save composite report
    out_report_path = os.path.join(output_dir, f"{base_name}_fracture_report.jpg")
    cv2.imwrite(out_report_path, composite)
    
    # Save single annotated image
    out_single_path = os.path.join(output_dir, f"{base_name}_detected.png")
    cv2.imwrite(out_single_path, header_canvas)
    
    print(f"Processed: {base_name:12} | Conf: {prob*100:5.1f}% | Subtype: {subtype_name[:22]:22} | Tilt: {metrics['angulation_deg']} deg | Gap: {metrics['displacement_gap_mm']} mm")
    return out_report_path, metrics

def batch_inference_all(model, device, input_dir=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\cleaned\unified_512"):
    output_dir = r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\predictions"
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(f"\nRunning Dual-View Orthopedic Inference & Subtype Analysis on {len(files)} radiographs...")
    print("=" * 95)
    
    reports = []
    for f in files:
        p = os.path.join(input_dir, f)
        report_path, metrics = run_diagnostic_inference(p, model, device, output_dir=output_dir)
        reports.append((f, report_path, metrics))
        
    print("=" * 95)
    print(f"All clinical orthopedic reports saved to: {output_dir}\n")
    return reports

if __name__ == "__main__":
    model, device = load_trained_model()
    batch_inference_all(model, device)

