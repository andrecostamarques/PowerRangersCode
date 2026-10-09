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
sys.stdout.reconfigure(line_buffering=True)

from SelectionMask import SelectionMask

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate a consensus mask from trained learnable masks.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="galaxy10",
        help="Target dataset name (e.g., galaxy10, food101, cifar10). Default is galaxy10."
    )
    parser.add_argument(
        "--checkpoints_dir", 
        type=str, 
        default=os.path.join(project_root, 'checkpoints'),
        help="Path to the checkpoints folder."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path where the consensus mask checkpoint will be saved. Defaults to checkpoints/consensus_mask_<dataset>.pt"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Binarization threshold for the averaged mask (0.0 to 1.0). Default is 0.5 (majority vote)."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Optional model name to exclude from consensus (e.g., resnet34)."
    )
    args = parser.parse_args()

    if args.output_path is None:
        if args.exclude:
            args.output_path = os.path.join(args.checkpoints_dir, f"consensus_mask_{args.dataset.lower()}_no_{args.exclude.lower()}.pt")
        else:
            args.output_path = os.path.join(args.checkpoints_dir, f"consensus_mask_{args.dataset.lower()}.pt")

    print(f"Project root: {project_root}")
    print(f"Checkpoints directory: {args.checkpoints_dir}")
    print(f"Output path: {args.output_path}")

    all_models = ["lenet256", "resnet20", "simplecnnrgb", "resnet34"]
    if args.exclude:
        all_models = [m for m in all_models if m.lower() != args.exclude.lower()]
        print(f"Excluding model '{args.exclude}'. Models included: {all_models}")

    folders = [f"{args.dataset.lower()}_{m}_200epochs" for m in all_models]

    bin_masks = []
    base_mask_model = None

    for model_name in all_models:
        folder_name = f"{args.dataset.lower()}_{model_name}_200epochs"
        possible_paths = [
            os.path.join(args.checkpoints_dir, folder_name),
            os.path.join(args.checkpoints_dir, args.dataset.lower(), folder_name),
            os.path.join(args.checkpoints_dir, args.dataset.lower(), model_name),
        ]
        
        folder_path = None
        for p in possible_paths:
            if os.path.exists(p):
                folder_path = p
                break

        # Se não encontrou nos caminhos padrões, faz busca recursiva na pasta de checkpoints
        if not folder_path and os.path.exists(args.checkpoints_dir):
            target_ds = args.dataset.lower()
            target_model = model_name.lower()
            for root, dirs, _ in os.walk(args.checkpoints_dir):
                for d in dirs:
                    d_lower = d.lower()
                    if (target_ds in root.lower() or target_ds in d_lower) and target_model in d_lower:
                        check_path = os.path.join(root, d)
                        if any(f.startswith("checkpoint_epoch_") for f in os.listdir(check_path)):
                            folder_path = check_path
                            break
                if folder_path:
                    break

        if not folder_path:
            print(f"Error: Directory for model '{model_name}' (dataset '{args.dataset}') not found in checkpoints hierarchy. Skipping.")
            continue

        # Encontra o último/melhor checkpoint ordenando por número de época
        pt_files = glob.glob(os.path.join(folder_path, "checkpoint_epoch_*.pt"))
        epochs_files = []
        for f in pt_files:
            match = re.search(r'checkpoint_epoch_(\d+)\.pt', f)
            if match:
                epochs_files.append((int(match.group(1)), f))

        epochs_files.sort(key=lambda x: x[0], reverse=True) # Maior número de época primeiro
        best_ckpt_file = epochs_files[0][1] if epochs_files else None

        if not best_ckpt_file:
            print(f"Warning: Could not find checkpoint files in {folder_path}.")
            continue

        print(f"Loading checkpoint for {folder_name} (Epoch {epochs_files[0][0]}) from: {best_ckpt_file}")
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

            if base_mask_model is None:
                base_mask_model = SelectionMask(shape=bin_mask.shape)
                print(f"Instantiated SelectionMask template with shape {bin_mask.shape}")

        except Exception as e:
            print(f"Error loading {best_ckpt_file}: {e}")
            continue

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

    ds_sub_path = os.path.join(args.checkpoints_dir, args.dataset.lower(), f"consensus_mask_{args.dataset.lower()}.pt")
    if os.path.abspath(ds_sub_path) != os.path.abspath(args.output_path):
        os.makedirs(os.path.dirname(ds_sub_path), exist_ok=True)
        torch.save(out_checkpoint, ds_sub_path)
        print(f"Also saved copy to: {ds_sub_path}")

if __name__ == "__main__":
    main()
