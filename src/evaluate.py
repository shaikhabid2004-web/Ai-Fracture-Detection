import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_fscore_support

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from src.model import DualBoneFractureNetwork
from src.dataset import BoneFractureDataset

def evaluate_pipeline(
    split="test",
    data_root=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data",
    model_path=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\checkpoints\best_fracture_model.pt",
    output_dashboard=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\clinical_pipeline_results.png",
    output_cm=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\confusion_matrix.png",
    output_json=r"c:\Users\shaik\OneDrive\Desktop\ai frcture anti\data\clinical_metrics.json"
):
    """
    Executes the clinical evaluation pipeline:
    Independent Test Set -> Model Prediction -> Confusion Matrix -> 
    Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBoneFractureNetwork(pretrained=False, num_classes=1)
    
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        epoch_info = ckpt.get('epoch', '?')
        val_acc_info = ckpt.get('val_accuracy', 0.0)
        print(f"Loaded model checkpoint from: {model_path} (Trained Epoch: {epoch_info}, Best Val Acc: {val_acc_info:.1f}%)")
    else:
        print(f"WARNING: Checkpoint {model_path} not found. Running with uninitialized weights.")
        
    model.to(device)
    model.eval()

    # Load Independent Test Set
    dataset = BoneFractureDataset(data_root=data_root, mode=split, is_train=False)
    print(f"Evaluating {len(dataset)} samples from Independent {split.upper()} Set...")
    
    y_true = []
    y_pred = []
    y_probs = []
    sample_records = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            inp = sample["image"].unsqueeze(0).to(device)
            target = int(sample["label"].item())
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = float(torch.sigmoid(logits).item())
            pred = 1 if prob >= 0.50 else 0
            
            y_true.append(target)
            y_pred.append(pred)
            y_probs.append(prob)
            sample_records.append({
                "filename": sample["filename"],
                "ground_truth": "Fracture" if target == 1 else "Normal",
                "predicted": "Fracture" if pred == 1 else "Normal",
                "probability": round(prob, 4),
                "correct": (pred == target)
            })

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_probs_np = np.array(y_probs)
    
    # 1. Confusion Matrix Elements
    tp = int(np.sum((y_pred_np == 1) & (y_true_np == 1)))
    fn = int(np.sum((y_pred_np == 0) & (y_true_np == 1)))
    fp = int(np.sum((y_pred_np == 1) & (y_true_np == 0)))
    tn = int(np.sum((y_pred_np == 0) & (y_true_np == 0)))
    total = len(y_true)
    
    # 2. Six Requested Clinical Metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate
    
    # ROC-AUC calculation
    if len(np.unique(y_true_np)) > 1:
        roc_auc = float(roc_auc_score(y_true_np, y_probs_np))
        fpr, tpr, thresholds = roc_curve(y_true_np, y_probs_np)
    else:
        roc_auc = 1.0 if accuracy == 1.0 else 0.5
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        
    metrics = {
        "dataset_split": split,
        "total_samples": total,
        "fracture_samples": int(np.sum(y_true_np == 1)),
        "normal_samples": int(np.sum(y_true_np == 0)),
        "confusion_matrix": {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "specificity": round(specificity, 4),
            "roc_auc": round(roc_auc, 4)
        },
        "sample_records": sample_records
    }
    
    # Print Beautiful Clinical CLI Summary
    print("\n" + "=" * 70)
    print(f"        CLINICAL ML EVALUATION REPORT ({split.upper()} SET)")
    print("=" * 70)
    print(f"Total Test Set Radiographs : {total} ({metrics['fracture_samples']} Fracture + {metrics['normal_samples']} Normal)")
    print(f"True Positives  (TP)       : {tp}")
    print(f"False Negatives (FN)       : {fn}")
    print(f"False Positives (FP)       : {fp}")
    print(f"True Negatives  (TN)       : {tn}")
    print("-" * 70)
    print(f" 1. Accuracy               : {accuracy * 100:.2f}%")
    print(f" 2. Precision              : {precision * 100:.2f}%")
    print(f" 3. Recall (Sensitivity)   : {recall * 100:.2f}%")
    print(f" 4. F1-Score               : {f1_score:.4f}")
    print(f" 5. Specificity            : {specificity * 100:.2f}%")
    print(f" 6. ROC-AUC                : {roc_auc:.4f}")
    print("=" * 70)
    
    # Save Metrics JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # Generate 3-Panel Comprehensive Clinical Dashboard Plot
    plt.style.use('default')
    fig, (ax_cm, ax_roc, ax_card) = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={'width_ratios': [1.1, 1.1, 1.2]})
    fig.patch.set_facecolor('#0b0f19') # Sleek dark medical background
    
    # PANEL 1: Confusion Matrix Heatmap
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    ax_cm.set_facecolor('#0b0f19')
    im = ax_cm.imshow(cm_matrix, cmap='Blues', interpolation='nearest')
    
    cell_annotations = [
        [f"{tn}\n\n[TN]\nNormal Intact", f"{fp}\n\n[FP]\nFalse Alarm"],
        [f"{fn}\n\n[FN]\nMissed Fracture", f"{tp}\n\n[TP]\nDetected Fracture"]
    ]
    
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            txt_color = "#ffffff" if val > 0 else "#64748b"
            ax_cm.text(j, i, cell_annotations[i][j], ha="center", va="center",
                       fontsize=10, fontweight="bold", color=txt_color)
                       
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Normal Bone', 'Fracture Bone'], fontsize=10, fontweight="bold", color="#cbd5e1")
    ax_cm.set_yticklabels(['Normal Bone', 'Fracture Bone'], fontsize=10, fontweight="bold", color="#cbd5e1")
    ax_cm.set_xlabel('Model Prediction', fontsize=11, fontweight="bold", color="#38bdf8", labelpad=8)
    ax_cm.set_ylabel('Ground Truth Diagnosis', fontsize=11, fontweight="bold", color="#38bdf8", labelpad=8)
    ax_cm.set_title(f'Confusion Matrix\n(Independent Test Set: N={total})', fontsize=12, fontweight="bold", color="#ffffff", pad=12)
    
    for spine in ax_cm.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.5)
        
    # PANEL 2: ROC Curve
    ax_roc.set_facecolor('#0f172a')
    ax_roc.plot(fpr, tpr, color='#38bdf8', lw=2.8, label=f'Model ROC (AUC = {roc_auc:.3f})')
    ax_roc.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--', label='Random Classifier (AUC = 0.50)')
    ax_roc.fill_between(fpr, tpr, color='#0284c7', alpha=0.25)
    
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight="bold", color="#cbd5e1")
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10, fontweight="bold", color="#cbd5e1")
    ax_roc.set_title(f'ROC Curve\nROC-AUC = {roc_auc:.4f}', fontsize=12, fontweight="bold", color="#ffffff", pad=12)
    ax_roc.tick_params(colors='#94a3b8')
    ax_roc.grid(True, linestyle=':', alpha=0.35, color='#475569')
    
    legend = ax_roc.legend(loc="lower right", facecolor='#1e293b', edgecolor='#334155', fontsize=9)
    for text in legend.get_texts():
        text.set_color("#f8fafc")
        
    for spine in ax_roc.spines.values():
        spine.set_color('#334155')
        spine.set_linewidth(1.5)
        
    # PANEL 3: 6 Metrics Dashboard Card
    ax_card.set_facecolor('#111827')
    ax_card.axis('off')
    
    metrics_display = [
        ("1. ACCURACY", f"{accuracy * 100:.1f}%", "#22c55e", "Correct / Total Decisions"),
        ("2. PRECISION", f"{precision * 100:.1f}%", "#38bdf8", "TP / (TP + FP)"),
        ("3. RECALL (SENSITIVITY)", f"{recall * 100:.1f}%", "#38bdf8", "TP / (TP + FN)"),
        ("4. F1-SCORE", f"{f1_score:.4f}", "#a855f7", "Harmonic Mean of P & R"),
        ("5. SPECIFICITY", f"{specificity * 100:.1f}%", "#f59e0b", "TN / (TN + FP) Healthy Correct"),
        ("6. ROC-AUC", f"{roc_auc:.4f}", "#ec4899", "Area Under ROC Curve"),
    ]
    
    ax_card.text(0.05, 0.94, "CLINICAL PERFORMANCE METRICS", fontsize=12, fontweight="bold", color="#ffffff", transform=ax_card.transAxes)
    ax_card.text(0.05, 0.88, f"Independent Test Set ({total} Radiographs)", fontsize=9, color="#94a3b8", transform=ax_card.transAxes)
    ax_card.plot([0.05, 0.95], [0.85, 0.85], color="#374151", transform=ax_card.transAxes, lw=1.5)
    
    y_pos = 0.76
    for title, val, col, sub in metrics_display:
        ax_card.text(0.05, y_pos, title, fontsize=9.5, fontweight="bold", color="#e2e8f0", transform=ax_card.transAxes)
        ax_card.text(0.05, y_pos - 0.04, sub, fontsize=7.5, color="#64748b", transform=ax_card.transAxes)
        ax_card.text(0.95, y_pos - 0.01, val, fontsize=12, fontweight="bold", color=col, ha="right", transform=ax_card.transAxes)
        y_pos -= 0.125
        
    plt.tight_layout()
    plt.savefig(output_dashboard, dpi=220, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(output_cm, dpi=220, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"\nSaved Comprehensive Dashboard to: {output_dashboard}")
    print(f"Saved Confusion Matrix to: {output_cm}")
    print(f"Saved Clinical Metrics JSON to: {output_json}\n")
    
    return metrics

if __name__ == "__main__":
    evaluate_pipeline(split="test")
