"""
Custom Dataset Module.

This module provides wrapper classes for external datasets to ensure compatibility
with PyTorch's Dataset interface and torchvision transformations. Specifically,
it handles the Galaxy10 DECals dataset from Hugging Face.
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

class Galaxy10HFDataset(Dataset):
    """
    Wraps the Galaxy10 DECals dataset from Hugging Face for PyTorch usage.

    This class handles downloading, caching, and loading the Galaxy10 dataset
    via the Hugging Face `datasets` library, while exposing an interface
    compatible with standard PyTorch DataLoaders (returning image tensors and labels).

    Attributes:
        dataset (datasets.ArrowBasedDataset): The underlying Hugging Face dataset object.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
    """

    def __init__(self, root, split='train', transform=None, download=True):
        """
        Initializes the Galaxy10HFDataset.

        Args:
            root (str): Root directory where the dataset cache will be stored.
            split (str, optional): The dataset split to load (e.g., 'train', 'test').
                Defaults to 'train'.
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. Defaults to None.
            download (bool, optional): Whether to download the dataset if not present.
                Defaults to True. (Note: `load_dataset` handles this automatically).
        """
        # Define the cache path within the root directory
        cache_path = os.path.join(root, 'hf_cache')
        
        print(f"Loading {split} split from Hugging Face (checking download/cache)...")
        
        # Load the dataset using Hugging Face's API.
        # This handles downloading and caching automatically.
        self.dataset = load_dataset(
            "matthieulel/galaxy10_decals", 
            split=split, 
            cache_dir=cache_path
        )
        
        self.transform = transform
        print(f"Galaxy10HFDataset: Loaded {len(self.dataset)} images for split '{split}'.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (image, label) where:
                - image (torch.Tensor): The transformed image tensor.
                - label (torch.Tensor): The class label as a LongTensor.
        """
        # Retrieve the item from the Hugging Face dataset
        # 'item' is a dictionary: {'image': <PIL.Image>, 'label': <int>}
        item = self.dataset[idx]
        
        image = item['image'] # PIL Image
        label = item['label'] # Integer label
        
        # Convert label to a torch tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply transformations (e.g., ToTensor, Normalize, Augmentations)
        if self.transform:
            image_tensor = self.transform(image)
        else:
            # Fallback: Convert to tensor if no transform is provided
            image_tensor = transforms.ToTensor()(image)

        # Return the tuple expected by PyTorch DataLoaders
        return image_tensor, label_tensor