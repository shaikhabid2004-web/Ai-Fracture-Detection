import os
import re
import csv
import cv2
import numpy as np
import shutil

def sanitize_filename(idx):
    return f"img_{idx:03d}"

def detect_banner_cutoff(img_gray):
    """
    Detects the row where the bottom metadata/text banner begins.
    Medical radiographs in this dataset contain patient/scan text
    printed in white on a dark bottom strip between 80% and 96% of the height.
    """
    h, w = img_gray.shape
    cutoff_row = h
    search_start = int(h * 0.80)
    
    for r in range(search_start, h):
        row_slice = img_gray[r, :]
        bright_count = np.sum(row_slice > 180)
        mean_intensity = np.mean(row_slice)
        # Condition: High contrast isolated text on dark background
        if bright_count > 5 and mean_intensity < 35:
            # Cut slightly above the first text line (with a 5-pixel safety margin)
            cutoff_row = max(search_start, r - 5)
            break
            
    return cutoff_row

def inpaint_peripheral_markers(img_gray):
    """
    Suppresses peripheral high-contrast anatomical orientation markers ('R' / 'L')
    that sit in the outer 8% margins on dark background.
    """
    ch, cw = img_gray.shape
    _, thresh = cv2.threshold(img_gray, 190, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    
    mask = np.zeros((ch, cw), dtype=np.uint8)
    for i in range(1, num_labels):
        x, y, sw, sh, area = stats[i]
        # Marker condition: in outer margins, character-sized, dark surrounding
        if (x < cw * 0.08 or x > cw * 0.92) and 8 <= sh <= 150 and 4 <= sw <= 120 and area < 6000:
            pad = 12
            y0, y1 = max(0, y - pad), min(ch, y + sh + pad)
            x0, x1 = max(0, x - pad), min(cw, x + sw + pad)
            local_mean = np.mean(img_gray[y0:y1, x0:x1])
            if local_mean < 75:
                mask[y:y+sh, x:x+sw] = 255

    if np.any(mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        cleaned = img_gray.copy()
        cleaned[mask_dilated > 0] = 0
        return cleaned
    return img_gray

def inpaint_metal_hardware(img_gray):
    """
    Detects and seamlessly inpaints orthopedic metal plates, screws, and surgical fixation strips.
    Metal hardware has the highest radiodensity (> 210) and distinct manufactured geometry.
    """
    h, w = img_gray.shape
    bone_pixels = img_gray[img_gray > 40]
    if len(bone_pixels) == 0:
        return img_gray, False
        
    p95 = np.percentile(bone_pixels, 95)
    thresh_val = max(205, int(p95))
    
    # 1. High brightness metal seeds
    metal_seeds = (img_gray >= thresh_val).astype(np.uint8) * 255
    
    # Exclude outer 6% margins (handled by marker inpainting)
    metal_seeds[:int(h*0.06), :] = 0
    metal_seeds[int(h*0.90):, :] = 0
    metal_seeds[:, :int(w*0.06)] = 0
    metal_seeds[:, int(w*0.94):] = 0
    
    # 2. Add high-contrast metal edges
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_norm = cv2.normalize(np.sqrt(sobelx**2 + sobely**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    metal_edges = ((grad_norm > 65) & (img_gray > 175)).astype(np.uint8) * 255
    
    combined = cv2.bitwise_or(metal_seeds, metal_edges)
    
    # 3. Morphological closing along vertical and horizontal plate axes
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 25))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 11))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_v)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_h)
    
    # 4. Filter connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    plate_mask = np.zeros((h, w), dtype=np.uint8)
    hardware_found = False
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:
            comp_mask = (labels == i)
            if np.mean(img_gray[comp_mask]) > 170:
                plate_mask[comp_mask] = 255
                hardware_found = True
                
    if hardware_found:
        # Fill holes inside metal plate mask (e.g. screw holes)
        cnts, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 120:
                cv2.drawContours(plate_mask, [c], -1, 255, thickness=cv2.FILLED)
                
        # Dilate for smooth boundary blending
        dilated = cv2.dilate(plate_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
        inpainted = cv2.inpaint(img_gray, dilated, inpaintRadius=11, flags=cv2.INPAINT_TELEA)
        return inpainted, True
        
    return img_gray, False

def apply_medical_clahe(img_gray, clip_limit=2.2, tile_size=(8, 8)):
    """
    Applies gentle bilateral filter followed by CLAHE to highlight
    fine bone trabeculae and fracture fissures without noise amplification.
    """
    denoised = cv2.bilateralFilter(img_gray, d=7, sigmaColor=35, sigmaSpace=35)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(denoised)

def letterbox_square(img_gray, target_size=1024):
    """
    Resizes image to target_size x target_size preserving exact aspect ratio.
    Uses seamless border-matched padding (median peripheral intensity) to eliminate
    artificial high-contrast border step-discontinuities.
    """
    ih, iw = img_gray.shape
    scale = min(target_size / iw, target_size / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Sample peripheral edges to determine true radiograph background tone
    border_pixels = np.concatenate([
        resized[0, :], resized[-1, :], resized[:, 0], resized[:, -1]
    ])
    pad_val = int(np.median(border_pixels))
    
    canvas = np.full((target_size, target_size), pad_val, dtype=np.uint8)
    dx = (target_size - nw) // 2
    dy = (target_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas

def split_dual_view(img_cropped):
    """
    If image is landscape (typical dual AP & Lateral projection),
    splits into Left (AP) and Right (Lateral/Oblique) views.
    """
    h, w = img_cropped.shape
    if w > h:
        mid = w // 2
        ap_view = img_cropped[:, :mid]
        lat_view = img_cropped[:, mid:]
        return ap_view, lat_view
    return None, None

def process_all_xrays(source_dir, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)
    raw_dir = os.path.join(output_base_dir, "raw")
    unified_1024_dir = os.path.join(output_base_dir, "cleaned", "unified_1024")
    unified_512_dir = os.path.join(output_base_dir, "cleaned", "unified_512")
    split_views_dir = os.path.join(output_base_dir, "cleaned", "split_views")
    
    for d in [raw_dir, unified_1024_dir, unified_512_dir, split_views_dir]:
        os.makedirs(d, exist_ok=True)
        
    files = sorted(
        [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999
    )
    
    print(f"Found {len(files)} X-ray images in {source_dir}")
    manifest_rows = []
    comparisons = []
    
    for f in files:
        src_path = os.path.join(source_dir, f)
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: Could not read {f}, skipping.")
            continue
            
        # Parse serial number (e.g., 1 from '1.jpg.jpeg')
        num_str = f.split('.')[0]
        idx = int(num_str) if num_str.isdigit() else len(manifest_rows) + 1
        base_name = sanitize_filename(idx)
        
        orig_h, orig_w = img.shape
        aspect = "landscape_dual" if orig_w > orig_h else "portrait_single"
        
        # 1. Copy raw file with standardized naming
        raw_dest = os.path.join(raw_dir, f"{base_name}.jpg")
        shutil.copy2(src_path, raw_dest)
        
        # 2. Detect & Crop Bottom Text Banner
        cutoff_row = detect_banner_cutoff(img)
        cropped = img[:cutoff_row, :]
        banner_removed = cutoff_row < orig_h
        
        # 3. Suppress peripheral markers ('R', 'L')
        marker_clean = inpaint_peripheral_markers(cropped)
        
        # 4. Inpaint and suppress surgical metal hardware (plates, screws, strips)
        metal_clean, metal_detected = inpaint_metal_hardware(marker_clean)
        
        # 5. Medical Contrast Enhancement (CLAHE) on metal-free bone
        enhanced = apply_medical_clahe(metal_clean)
        
        # 5. Letterbox to Square Resolutions
        sq_1024 = letterbox_square(enhanced, target_size=1024)
        sq_512 = letterbox_square(enhanced, target_size=512)
        
        out_1024_path = os.path.join(unified_1024_dir, f"{base_name}.png")
        out_512_path = os.path.join(unified_512_dir, f"{base_name}.png")
        cv2.imwrite(out_1024_path, sq_1024)
        cv2.imwrite(out_512_path, sq_512)
        
        # 6. Split Dual Views (if landscape)
        split_created = False
        ap_view, lat_view = split_dual_view(enhanced)
        if ap_view is not None and lat_view is not None:
            split_created = True
            ap_sq = letterbox_square(ap_view, target_size=512)
            lat_sq = letterbox_square(lat_view, target_size=512)
            cv2.imwrite(os.path.join(split_views_dir, f"{base_name}_AP.png"), ap_sq)
            cv2.imwrite(os.path.join(split_views_dir, f"{base_name}_LAT.png"), lat_sq)
            
        manifest_rows.append({
            "image_id": base_name,
            "original_filename": f,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "aspect_type": aspect,
            "banner_cutoff_row": cutoff_row,
            "banner_removed": banner_removed,
            "metal_hardware_removed": metal_detected,
            "split_dual_views": split_created,
            "has_fracture": True
        })
        
        # Collect samples for verification contact sheet (first 4 images)
        if len(comparisons) < 4:
            # Resize raw to 512 for visual comparison
            raw_sq = letterbox_square(img, target_size=512)
            comparisons.append((base_name, raw_sq, sq_512))
            
        print(f"Processed: {f:12} -> {base_name}.png [Crop: {cutoff_row}/{orig_h} | Metal Removed: {metal_detected} | Aspect: {aspect}]")
        
    # Write Manifest CSV
    manifest_csv = os.path.join(output_base_dir, "dataset_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "image_id", "original_filename", "orig_width", "orig_height",
            "aspect_type", "banner_cutoff_row", "banner_removed", "metal_hardware_removed", "split_dual_views", "has_fracture"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow(r)
            
    print(f"\nManifest saved to: {manifest_csv}")
    
    # Generate Cleaning Verification Contact Sheet (Before vs After)
    if comparisons:
        rows = []
        for name, before, after in comparisons:
            # Add header text
            cv2.putText(before, f"RAW: {name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(after, f"CLEANED + CLAHE: {name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            row = np.hstack([before, after])
            rows.append(row)
        sheet = np.vstack(rows)
        sheet_path = os.path.join(output_base_dir, "cleaning_verification.jpg")
        cv2.imwrite(sheet_path, sheet)
        print(f"Saved cleaning verification contact sheet to: {sheet_path}")
        
    print(f"\nAll {len(manifest_rows)} X-rays successfully cleaned and standardized!")
    return manifest_rows

if __name__ == "__main__":
    SOURCE_DIR = r"C:\Users\shaik\OneDrive\Desktop\xray"
    OUTPUT_BASE = r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data"
    process_all_xrays(SOURCE_DIR, OUTPUT_BASE)
