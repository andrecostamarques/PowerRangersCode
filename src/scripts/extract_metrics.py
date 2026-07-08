import os
import sys
import glob
import re
import torch
import numpy as np

# Add project directories to sys.path to allow unpickling SelectionMask
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/models'))
sys.path.append(os.path.join(project_root, 'src/utils'))

def calculate_metrics(cm):
    eps = 1e-9
    cm = np.array(cm)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * (p * r) / (p + r + eps)
    
    precision = np.mean(p) * 100
    recall = np.mean(r) * 100
    f1_score = np.mean(f1) * 100
    return precision, recall, f1_score

def get_active_pixels_ratio(checkpoint):
    # Try to get active pixels ratio from the checkpoint mask weights
    if 'mask_state_dict' in checkpoint and 'mask' in checkpoint['mask_state_dict']:
        mask_weight = checkpoint['mask_state_dict']['mask']
        bin_mask = torch.round(torch.sigmoid(mask_weight)).float()
        active_pixels = (bin_mask == 1.0).sum().item()
        total_pixels = bin_mask.numel()
        return (active_pixels / total_pixels) * 100
    return None

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    checkpoints_dir = os.path.join(project_root, 'checkpoints')
    
    models = ["lenet256", "resnet20", "resnet34", "simplecnnrgb"]
    regimes = {
        "normal": "Baseline",
        "200epochs": "Learnable Mask",
        "consensus_mask": "Consensus Mask"
    }
    
    results = []
    
    for model in models:
        for suffix, regime_name in regimes.items():
            folder_name = f"galaxy10_{model}_{suffix}"
            folder_path = os.path.join(checkpoints_dir, folder_name)
            
            if not os.path.exists(folder_path):
                results.append({
                    "model": model,
                    "regime": regime_name,
                    "status": "Not Found",
                    "best_epoch": "-",
                    "accuracy": "-",
                    "precision": "-",
                    "recall": "-",
                    "f1_score": "-",
                    "sparsity": "-"
                })
                continue
                
            # Find the best checkpoint (checkpoint_epoch_X.pt where X != 1)
            pt_files = glob.glob(os.path.join(folder_path, "checkpoint_epoch_*.pt"))
            best_ckpt_file = None
            best_epoch = None
            for f in pt_files:
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', f)
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num != 1:
                        best_ckpt_file = f
                        best_epoch = epoch_num
                        break
            
            # Fallback if only epoch 1 exists or if structure is different
            if not best_ckpt_file and os.path.exists(os.path.join(folder_path, "checkpoint_epoch_1.pt")):
                best_ckpt_file = os.path.join(folder_path, "checkpoint_epoch_1.pt")
                best_epoch = 1
                
            if not best_ckpt_file:
                results.append({
                    "model": model,
                    "regime": regime_name,
                    "status": "No Checkpoint",
                    "best_epoch": "-",
                    "accuracy": "-",
                    "precision": "-",
                    "recall": "-",
                    "f1_score": "-",
                    "sparsity": "-"
                })
                continue
                
            try:
                checkpoint = torch.load(best_ckpt_file, map_location='cpu', weights_only=False)
                
                # Accuracy
                acc = checkpoint.get('accuracy', checkpoint.get('val_accuracy', None))
                
                # Confusion matrix metrics
                precision_val, recall_val, f1_val = "-", "-", "-"
                if 'cm' in checkpoint:
                    p, r, f1 = calculate_metrics(checkpoint['cm'])
                    precision_val = f"{p:.2f}%"
                    recall_val = f"{r:.2f}%"
                    f1_val = f"{f1:.2f}%"
                
                # Sparsity (% of active pixels)
                sparsity_val = "100.00%"
                if suffix == "200epochs":
                    ratio = get_active_pixels_ratio(checkpoint)
                    if ratio is not None:
                        sparsity_val = f"{ratio:.2f}%"
                    else:
                        sparsity_val = "Error"
                elif suffix == "consensus_mask":
                    # Load consensus_mask.pt to get its sparsity
                    consensus_path = os.path.join(checkpoints_dir, 'consensus_mask.pt')
                    if os.path.exists(consensus_path):
                        cons_ckpt = torch.load(consensus_path, map_location='cpu', weights_only=False)
                        ratio = get_active_pixels_ratio(cons_ckpt)
                        if ratio is not None:
                            sparsity_val = f"{ratio:.2f}%"
                        else:
                            sparsity_val = "Error"
                    else:
                        sparsity_val = "Missing consensus_mask.pt"
                
                results.append({
                    "model": model,
                    "regime": regime_name,
                    "status": "OK",
                    "best_epoch": best_epoch,
                    "accuracy": f"{acc:.2f}%" if acc is not None else "-",
                    "precision": precision_val,
                    "recall": recall_val,
                    "f1_score": f1_val,
                    "sparsity": sparsity_val
                })
                
            except Exception as e:
                results.append({
                    "model": model,
                    "regime": regime_name,
                    "status": f"Error: {str(e)}",
                    "best_epoch": best_epoch if best_epoch else "-",
                    "accuracy": "-",
                    "precision": "-",
                    "recall": "-",
                    "f1_score": "-",
                    "sparsity": "-"
                })
                
    # Print the markdown table
    print("# Galaxy10 Model Performance Comparison\n")
    print("| Model Architecture | Training Regime | Best Epoch | Val Accuracy | Precision | Recall | F1-Score | Active Pixels (Sparsity) | Status |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(f"| {r['model']} | {r['regime']} | {r['best_epoch']} | {r['accuracy']} | {r['precision']} | {r['recall']} | {r['f1_score']} | {r['sparsity']} | {r['status']} |")

if __name__ == "__main__":
    main()
