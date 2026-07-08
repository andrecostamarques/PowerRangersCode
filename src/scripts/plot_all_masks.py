import os
import sys
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project directories to sys.path to allow unpickling SelectionMask
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/models'))
sys.path.append(os.path.join(project_root, 'src/utils'))

def get_mask(model_name, regime, checkpoints_dir):
    if regime == "normal":
        # Baseline has no mask (100% active)
        return np.ones((3, 256, 256))
        
    if regime == "consensus_mask":
        consensus_path = os.path.join(checkpoints_dir, 'consensus_mask.pt')
        if os.path.exists(consensus_path):
            try:
                data = torch.load(consensus_path, map_location='cpu', weights_only=False)
                mask_weights = data['mask_state_dict']['mask'].squeeze().cpu()
                return torch.round(torch.sigmoid(mask_weights)).numpy()
            except Exception as e:
                print(f"Error loading consensus mask: {e}")
                return np.ones((3, 256, 256))
        else:
            return np.ones((3, 256, 256))
            
    # For learnable mask (200epochs)
    folder_name = f"galaxy10_{model_name}_200epochs"
    folder_path = os.path.join(checkpoints_dir, folder_name)
    if os.path.exists(folder_path):
        pt_files = glob.glob(os.path.join(folder_path, "checkpoint_epoch_*.pt"))
        best_ckpt = None
        for f in pt_files:
            match = re.search(r'checkpoint_epoch_(\d+)\.pt', f)
            if match and int(match.group(1)) != 1:
                best_ckpt = f
                break
        if best_ckpt:
            try:
                data = torch.load(best_ckpt, map_location='cpu', weights_only=False)
                if 'mask_state_dict' in data:
                    mask_weights = data['mask_state_dict']['mask'].squeeze().cpu()
                    return torch.round(torch.sigmoid(mask_weights)).numpy()
            except Exception as e:
                print(f"Error loading learnable mask for {model_name}: {e}")
    return np.zeros((3, 256, 256)) # Return black if not found

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    checkpoints_dir = os.path.join(project_root, 'checkpoints')
    
    models = ["lenet256", "resnet20", "resnet34", "simplecnnrgb"]
    regimes = ["normal", "200epochs", "consensus_mask"]
    regime_labels = ["Baseline", "Learnable Mask", "Consensus Mask"]
    
    # 3 rows (regimes) x 4 columns (models)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row_idx, regime in enumerate(regimes):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            mask = get_mask(model, regime, checkpoints_dir)
            
            # Combine the 3 channels (RGB) into a single displayable image
            # Transpose from (3, 256, 256) to (256, 256, 3)
            rgb_mask = np.transpose(mask, (1, 2, 0))
            
            ax.imshow(rgb_mask, origin='lower')
            
            # Titles and Labels
            if row_idx == 0:
                ax.set_title(model.upper(), fontsize=14, fontweight='bold', pad=10)
            if col_idx == 0:
                ax.set_ylabel(regime_labels[row_idx], fontsize=14, fontweight='bold', labelpad=15)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.axis('off')
            
    plt.tight_layout()
    output_image = os.path.join(checkpoints_dir, 'all_masks_comparison.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Comparison plot successfully saved to: {output_image}")

if __name__ == "__main__":
    main()
