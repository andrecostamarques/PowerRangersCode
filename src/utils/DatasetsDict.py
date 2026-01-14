"""
Dataset dictionary management module.

This module provides a centralized repository for accessing various standard datasets
(MNIST, FashionMNIST, CIFAR-10, SVHN) with predefined transformations for training
and testing.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from CustomDatasets import Galaxy10HFDataset


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

        # =========================================================================
        # 5. Galaxy10
        # =========================================================================

        galaxy10_mean = (0.170, 0.142, 0.121)
        galaxy10_std  = (0.248, 0.210, 0.190)

        galaxy10_tf_train = transforms.Compose([
            transforms.RandomCrop(256, padding=32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(galaxy10_mean, galaxy10_std),
        ])

        galaxy10_tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(galaxy10_mean, galaxy10_std),
        ])

        # Seguindo o seu padrão de retornar [train_ds, test_ds]
        galaxy10_ds = [
            Galaxy10HFDataset(root=f'{self.data_root}galaxy10', split='train', download=True, transform=galaxy10_tf_train),
            Galaxy10HFDataset(root=f'{self.data_root}galaxy10', split='test', download=True, transform=galaxy10_tf_test)
        ]
        
        self.datasets['galaxy10'] = (galaxy10_ds, galaxy10_tf_train, galaxy10_tf_test)

        # =========================================================================
        # 6. EuroSAT
        # =========================================================================
        # EuroSAT é 256x256 por padrão no seu teste com LeNet
        eurosat_mean = (0.485, 0.456, 0.406)
        eurosat_std  = (0.229, 0.224, 0.225)

        eurosat_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(eurosat_mean, eurosat_std),
        ])

        # O EuroSAT não possui split='train' ou 'test' nativo no torchvision
        # Carregamos o dataset completo e dividimos manualmente
        full_eurosat = datasets.EuroSAT(
            root=f'{self.data_root}eurosat', 
            download=True, 
            transform=eurosat_tf
        )

        # Divisão 80% treino, 20% teste
        train_len = int(0.8 * len(full_eurosat))
        test_len = len(full_eurosat) - train_len
        
        # Gerando os objetos de dataset
        eurosat_train, eurosat_test = random_split(
            full_eurosat, 
            [train_len, test_len],
            generator=torch.Generator().manual_seed(42) # Seed fixa para reprodutibilidade
        )

        self.datasets['eurosat'] = ([eurosat_train, eurosat_test], eurosat_tf, eurosat_tf)

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