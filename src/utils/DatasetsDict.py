"""
Dataset dictionary management module.

This module provides a centralized repository for accessing various standard datasets
(MNIST, FashionMNIST, CIFAR-10, SVHN) with predefined transformations for training
and testing.
"""

import torch
from torchvision import datasets, transforms

class DatasetDict:
    """
    A dictionary-like container for managing and retrieving datasets.

    This class initializes and stores configurations for multiple datasets, including
    their training and testing splits and associated transformations.

    Attributes:
        datasets (dict): A dictionary storing dataset configurations. Keys are dataset names
            (e.g., 'mnist', 'cifar10'), and values are tuples containing:
            ([train_dataset, test_dataset], transform_train, transform_test).
        data_root (str): The root directory where datasets are stored or downloaded.
    """

    def __init__(self, root_dir_save = '../../data/'):
        """
        Initializes the DatasetDict.

        Args:
            root_dir_save (str, optional): The root directory path where datasets will be
                downloaded and saved. Defaults to '../../data/'.
        """
        # Each key will have:( [train_ds, test_ds], transform_train, transform_test )
        
        self.datasets = {}
        self.data_root = root_dir_save
        self._initialize_datasets()

    def _initialize_datasets(self):
        """
        Initializes the supported datasets with specific transformations.

        Configures the following datasets:
        - MNIST
        - FashionMNIST
        - CIFAR-10
        - SVHN

        Each dataset is configured with appropriate normalization and augmentation
        (for training sets where applicable) and stored in the `self.datasets` dictionary.
        """

        # =========================================================================
        # 1. MNIST
        # =========================================================================

        # Each dataset needs to have three things:
        # The transforms and the Dataset object
        # With that, we can have the dataloader

        mnist_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        mnist_ds = [
            datasets.MNIST(root=self.data_root, train=True, download=True, transform=mnist_tf),
            datasets.MNIST(root=self.data_root, train=False, download=True, transform=mnist_tf)
        ]
        self.datasets['mnist'] = (mnist_ds, mnist_tf, mnist_tf)

        # =========================================================================
        # 2. FashionMNIST
        # =========================================================================
        fmnist_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

        fmnist_ds = [
            datasets.FashionMNIST(root=self.data_root, train=True, download=True, transform=fmnist_tf),
            datasets.FashionMNIST(root=self.data_root, train=False, download=True, transform=fmnist_tf)
        ]
        self.datasets['fmnist'] = (fmnist_ds, fmnist_tf, fmnist_tf)

        # =========================================================================
        # 3. CIFAR-10
        # =========================================================================
        cifar_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        
        cifar_tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            cifar_norm,
        ])

        cifar_tf_test = transforms.Compose([
            transforms.ToTensor(),
            cifar_norm,
        ])

        cifar_ds = [
            datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=cifar_tf_train),
            datasets.CIFAR10(root=self.data_root, train=False, download=True, transform=cifar_tf_test)
        ]
        self.datasets['cifar10'] = (cifar_ds, cifar_tf_train, cifar_tf_test)

        # =========================================================================
        # 4. SVHN
        # =========================================================================
        svhn_mean = (0.4377, 0.4438, 0.4728)
        svhn_std  = (0.1980, 0.2010, 0.1970)

        svhn_tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])

        svhn_tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(svhn_mean, svhn_std),
        ])

        svhn_ds = [
            datasets.SVHN(root=f'{self.data_root}svhn', split='train', download=True, transform=svhn_tf_train),
            datasets.SVHN(root=f'{self.data_root}svhn', split='test', download=True, transform=svhn_tf_test)
        ]
        self.datasets['svhn'] = (svhn_ds, svhn_tf_train, svhn_tf_test)

    def get(self, name):
        """
        Retrieves a dataset configuration by name.

        Args:
            name (str): The name of the dataset to retrieve (case-insensitive).
                Supported names: 'mnist', 'fmnist', 'cifar10', 'svhn'.

        Returns:
            tuple: A tuple containing:
                - list: [train_dataset, test_dataset] objects.
                - torchvision.transforms.Compose: Transformations for the training set.
                - torchvision.transforms.Compose: Transformations for the test set.

        Raises:
            ValueError: If the requested dataset name is not found.
        """
        # It returns the object in the dictionary
        name = name.lower()
        if name in self.datasets:
            return self.datasets[name]
        else:
            raise ValueError(f"Dataset '{name}' not found.")