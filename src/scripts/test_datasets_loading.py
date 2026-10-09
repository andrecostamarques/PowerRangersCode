import sys
import os

project_root = "/home/andre-marques/Desktop/Estudo/PowerRangersCode"
sys.path.append(os.path.join(project_root, "src/utils"))
sys.path.append(os.path.join(project_root, "src/models"))

from DatasetsDict import DatasetDict

print("Initializing DatasetDict...")
dd = DatasetDict(root_dir_save=os.path.join(project_root, "data/"))

print("Supported datasets:", list(dd.datasets.keys()))

try:
    cifar10_data = dd.get('cifar10')
    print("CIFAR-10 loaded successfully! Train samples:", len(cifar10_data[0][0]))
except Exception as e:
    print("Error loading CIFAR-10:", e)

try:
    food101_data = dd.get('food101')
    print("Food101 loaded successfully! Train samples:", len(food101_data[0][0]))
except Exception as e:
    print("Error loading Food101:", e)
