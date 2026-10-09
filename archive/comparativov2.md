| Model Architecture | Training Regime | Best Epoch | Val Accuracy | Precision | Recall | F1-Score | Active Pixels (Sparsity) | Status |
| --------------------| -----------------| ------------| --------------| -----------| --------| ----------| --------------------------| --------|
| lenet256           | Baseline        | 168        | 75.44%       | 75.41%    | 72.79% | 73.62%   | 100.00%                  | OK     |
| lenet256           | Learnable Mask  | 164        | 74.56%       | 74.05%    | 71.52% | 71.64%   | 9.77%                    | OK     |
| lenet256           | Consensus Mask  | 192        | 74.81%       | 74.31%    | 70.89% | 71.72%   | 10.11%                   | OK     |
| resnet20           | Baseline        | 184        | 84.46%       | 84.28%    | 82.60% | 83.26%   | 100.00%                  | OK     |
| resnet20           | Learnable Mask  | 159        | 84.59%       | 83.79%    | 81.66% | 82.49%   | 32.70%                   | OK     |
| resnet20           | Consensus Mask  | 168        | 82.27%       | 80.74%    | 80.24% | 80.24%   | 10.11%                   | OK     |
| resnet34           | Baseline        | 72         | 85.65%       | 84.02%    | 84.38% | 84.06%   | 100.00%                  | OK     |
| resnet34           | Learnable Mask  | 223        | 70.74%       | 69.88%    | 68.16% | 68.07%   | 0.19%                    | OK     |
| resnet34           | Consensus Mask  | 162        | 85.59%       | 85.19%    | 83.81% | 84.43%   | 10.11%                   | OK     |
| simplecnnrgb       | Baseline        | 190        | 80.51%       | 80.45%    | 77.42% | 78.36%   | 100.00%                  | OK     |
| simplecnnrgb       | Learnable Mask  | 200        | 74.75%       | 73.78%    | 70.34% | 70.89%   | 8.97%                    | OK     |
| simplecnnrgb       | Consensus Mask  | 198        | 77.88%       | 78.40%    | 75.04% | 74.96%   | 10.11%                   | OK     |
