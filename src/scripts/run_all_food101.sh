#!/usr/bin/env bash
set -e

# Navigate to project root directory regardless of invocation location
cd "$(dirname "$0")/../.."

echo "============================================================"
echo "STARTING FOOD-101 FULL EXPERIMENTAL PIPELINE (200 EPOCHS)"
echo "============================================================"

# 1. Baseline Training (no mask)
echo "--- Step 1: Baseline Training ---"
python3 src/scripts/train_classifier.py --dataset food101 --model lenet256 --epochs 200
python3 src/scripts/train_classifier.py --dataset food101 --model resnet20 --epochs 200
python3 src/scripts/train_classifier.py --dataset food101 --model simplecnnrgb --epochs 200
python3 src/scripts/train_classifier.py --dataset food101 --model resnet34 --epochs 200

# 2. Learnable Mask Training (joint model + selection mask)
echo "--- Step 2: Learnable Mask Training ---"
python3 src/scripts/run_experiments.py --dataset food101 --model all --epochs 200

# 3. Generate Consensus Mask
echo "--- Step 3: Generating Consensus Mask ---"
python3 src/scripts/generate_consensus_mask.py --dataset food101

# 4. Consensus Mask Retraining (frozen consensus mask)
echo "--- Step 4: Consensus Mask Retraining ---"
python3 src/scripts/train_classifier.py --dataset food101 --model lenet256 --mask_checkpoint checkpoints/consensus_mask_food101.pt --epochs 200
python3 src/scripts/train_classifier.py --dataset food101 --model resnet20 --mask_checkpoint checkpoints/consensus_mask_food101.pt --epochs 200
python3 src/scripts/train_classifier.py --dataset food101 --model simplecnnrgb --mask_checkpoint checkpoints/consensus_mask_food101.pt --epochs 200
python3 src/scripts/train_classifier.py --dataset food101 --model resnet34 --mask_checkpoint checkpoints/consensus_mask_food101.pt --epochs 200

echo "============================================================"
echo "FOOD-101 PIPELINE COMPLETED SUCCESSFULLY!"
echo "============================================================"
