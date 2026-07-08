#!/usr/bin/env python3
import os
import sys
import glob
import re
import torch

# Setup path to import SelectionMask
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(project_root, 'src/models'))
sys.path.append(os.path.join(project_root, 'src/utils'))

from SelectionMask import SelectionMask

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate a consensus mask from the best checkpoints of the 4 models.")
    parser.add_argument(
        "--checkpoints_dir", 
        type=str, 
        default=os.path.join(project_root, 'checkpoints'),
        help="Path to the checkpoints folder."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=os.path.join(project_root, 'checkpoints/consensus_mask.pt'),
        help="Path where the consensus mask checkpoint will be saved."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Binarization threshold for the averaged mask (0.0 to 1.0). Default is 0.5 (majority vote)."
    )
    args = parser.parse_args()

    print(f"Project root: {project_root}")
    print(f"Checkpoints directory: {args.checkpoints_dir}")

    # List of the 4 folders
    folders = [
        "galaxy10_lenet256_200epochs",
        "galaxy10_resnet20_200epochs",
        "galaxy10_resnet34_200epochs",
        "galaxy10_simplecnnrgb_200epochs"
    ]

    bin_masks = []
    base_mask_model = None

    for folder_name in folders:
        folder_path = os.path.join(args.checkpoints_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Error: Directory {folder_path} does not exist. Skipping.")
            continue

        # Find the best checkpoint (checkpoint_epoch_X.pt where X != 1)
        pt_files = glob.glob(os.path.join(folder_path, "checkpoint_epoch_*.pt"))
        best_ckpt_file = None
        for f in pt_files:
            match = re.search(r'checkpoint_epoch_(\d+)\.pt', f)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num != 1:
                    best_ckpt_file = f
                    break

        if not best_ckpt_file:
            print(f"Warning: Could not find best checkpoint (epoch != 1) in {folder_path}.")
            continue

        print(f"Loading best checkpoint for {folder_name} from: {best_ckpt_file}")
        try:
            checkpoint = torch.load(best_ckpt_file, map_location='cpu', weights_only=False)
            if 'mask_state_dict' not in checkpoint:
                print(f"Warning: 'mask_state_dict' not found in {best_ckpt_file}. Skipping.")
                continue

            mask_weight = checkpoint['mask_state_dict']['mask']
            # Compute binary mask: round(sigmoid(weight))
            bin_mask = torch.round(torch.sigmoid(mask_weight)).float()
            bin_masks.append(bin_mask)
            
            # Keep track of active pixel ratio of this individual mask
            active_pixels = (bin_mask == 1.0).sum().item()
            total_pixels = bin_mask.numel()
            ratio = active_pixels / total_pixels
            print(f"  -> Active pixels: {active_pixels}/{total_pixels} ({ratio * 100:.2f}%)")

        except Exception as e:
            print(f"Error loading {best_ckpt_file}: {e}")
            continue

        # Load epoch 1 checkpoint to extract the mask_model_obj structure if not done yet
        if base_mask_model is None:
            epoch1_path = os.path.join(folder_path, "checkpoint_epoch_1.pt")
            if os.path.exists(epoch1_path):
                try:
                    checkpoint_e1 = torch.load(epoch1_path, map_location='cpu', weights_only=False)
                    if 'mask_model_obj' in checkpoint_e1:
                        base_mask_model = checkpoint_e1['mask_model_obj']
                        print(f"Loaded base SelectionMask object template from {epoch1_path}")
                except Exception as e:
                    print(f"Warning: Failed to load epoch 1 template from {epoch1_path}: {e}")

    if not bin_masks:
        print("Error: No binary masks could be loaded. Cannot generate consensus mask.")
        return

    # Average the binary masks
    print(f"Averaging {len(bin_masks)} masks...")
    avg_bin_mask = sum(bin_masks) / len(bin_masks)

    # Apply binarization threshold (e.g. >= 0.5)
    consensus_bin_mask = (avg_bin_mask >= args.threshold).float()
    
    # Calculate active pixel ratio of the consensus mask
    active_pixels = (consensus_bin_mask == 1.0).sum().item()
    total_pixels = consensus_bin_mask.numel()
    ratio = active_pixels / total_pixels
    print(f"Consensus Mask Active Pixels (threshold={args.threshold}): {active_pixels}/{total_pixels} ({ratio * 100:.2f}%)")

    # If we couldn't load the object from epoch 1, instantiate a new one
    if base_mask_model is None:
        print("Warning: Could not load base mask object template from any checkpoint. Instantiating a new SelectionMask(shape=(3, 256, 256)).")
        base_mask_model = SelectionMask(shape=(3, 256, 256))

    # Set weights to make it represent the consensus mask exactly
    # sigmoid(10.0) rounds to 1.0, sigmoid(-10.0) rounds to 0.0
    new_mask_weights = torch.where(consensus_bin_mask == 1.0, torch.tensor(10.0), torch.tensor(-10.0))
    base_mask_model.mask.data = new_mask_weights

    # Save to path
    out_checkpoint = {
        'mask_model_obj': base_mask_model,
        'mask_state_dict': base_mask_model.state_dict()
    }
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(out_checkpoint, args.output_path)
    print(f"Consensus mask successfully generated and saved to: {args.output_path}")

if __name__ == "__main__":
    main()
