import os
import torch
import torchvision.datasets as datasets
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def verify_dataset_integrity(dataset_name: str, data_root: str = './data') -> bool:
    """
    Verify that a dataset has been downloaded and can be loaded successfully.
    
    Args:
        dataset_name: Name of the dataset to verify ('mnist', 'svhn', 'cifar10', 'cifar100')
        data_root: Root directory for datasets
    
    Returns:
        bool: True if dataset is valid and accessible, False otherwise
    
    Raises:
        ValueError: If dataset_name is not recognized
    """
    dataset_name = dataset_name.lower()
    
    logger.info(f"Verifying integrity of {dataset_name} dataset...")
    
    try:
        # Attempt to load the dataset
        if dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root=f'{data_root}/raw',
                train=True,
                download=False
            )
            test_dataset = datasets.MNIST(
                root=f'{data_root}/raw',
                train=False,
                download=False
            )
            expected_train_size = 60000
            expected_test_size = 10000
            
        elif dataset_name == 'svhn':
            train_dataset = datasets.SVHN(
                root=f'{data_root}/raw',
                split='train',
                download=False
            )
            test_dataset = datasets.SVHN(
                root=f'{data_root}/raw',
                split='test',
                download=False
            )
            expected_train_size = 73257
            expected_test_size = 26032
            
        elif dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=f'{data_root}/raw',
                train=True,
                download=False
            )
            test_dataset = datasets.CIFAR10(
                root=f'{data_root}/raw',
                train=False,
                download=False
            )
            expected_train_size = 50000
            expected_test_size = 10000
            
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(
                root=f'{data_root}/raw',
                train=True,
                download=False
            )
            test_dataset = datasets.CIFAR100(
                root=f'{data_root}/raw',
                train=False,
                download=False
            )
            expected_train_size = 50000
            expected_test_size = 10000
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Supported datasets: mnist, svhn, cifar10, cifar100")
        
        # Check dataset sizes
        train_size = len(train_dataset)
        test_size = len(test_dataset)
        
        logger.info(f"Train set size: {train_size} (expected: {expected_train_size})")
        logger.info(f"Test set size: {test_size} (expected: {expected_test_size})")
        
        if train_size != expected_train_size:
            logger.warning(f"Train set size mismatch for {dataset_name}: "
                         f"got {train_size}, expected {expected_train_size}")
            return False
            
        if test_size != expected_test_size:
            logger.warning(f"Test set size mismatch for {dataset_name}: "
                         f"got {test_size}, expected {expected_test_size}")
            return False
        
        # Try to load a sample
        sample = train_dataset[0]
        if sample is None or len(sample) < 2:
            logger.error(f"Failed to load sample from {dataset_name}")
            return False
        
        logger.info(f"✓ {dataset_name} dataset integrity verified successfully!")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found for {dataset_name}: {e}")
        logger.info(f"Please download the dataset first using: "
                   f"bash scripts/download_datasets.sh")
        return False
        
    except Exception as e:
        logger.error(f"Error verifying {dataset_name} dataset: {e}")
        return False


def get_dataset_stats(dataset_name: str, data_root: str = './data') -> Dict:
    """
    Get statistics about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_root: Root directory for datasets
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    dataset_name = dataset_name.lower()
    
    stats = {
        'name': dataset_name,
        'train_size': 0,
        'test_size': 0,
        'num_classes': 0,
        'image_size': None,
        'num_channels': 0
    }
    
    try:
        if dataset_name == 'mnist':
            train_dataset = datasets.MNIST(root=f'{data_root}/raw', train=True, download=False)
            test_dataset = datasets.MNIST(root=f'{data_root}/raw', train=False, download=False)
            stats['num_classes'] = 10
            stats['image_size'] = (28, 28)
            stats['num_channels'] = 1
            
        elif dataset_name == 'svhn':
            train_dataset = datasets.SVHN(root=f'{data_root}/raw', split='train', download=False)
            test_dataset = datasets.SVHN(root=f'{data_root}/raw', split='test', download=False)
            stats['num_classes'] = 10
            stats['image_size'] = (32, 32)
            stats['num_channels'] = 3
            
        elif dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(root=f'{data_root}/raw', train=True, download=False)
            test_dataset = datasets.CIFAR10(root=f'{data_root}/raw', train=False, download=False)
            stats['num_classes'] = 10
            stats['image_size'] = (32, 32)
            stats['num_channels'] = 3
            
        elif dataset_name == 'cifar100':
            train_dataset = datasets.CIFAR100(root=f'{data_root}/raw', train=True, download=False)
            test_dataset = datasets.CIFAR100(root=f'{data_root}/raw', train=False, download=False)
            stats['num_classes'] = 100
            stats['image_size'] = (32, 32)
            stats['num_channels'] = 3
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        stats['train_size'] = len(train_dataset)
        stats['test_size'] = len(test_dataset)
        
    except Exception as e:
        logger.error(f"Error getting stats for {dataset_name}: {e}")
    
    return stats


def download_dataset(dataset_name: str, data_root: str = './data') -> bool:
    """
    Download a dataset if not already present.
    
    Args:
        dataset_name: Name of the dataset to download
        data_root: Root directory for datasets
    
    Returns:
        bool: True if download successful, False otherwise
    """
    dataset_name = dataset_name.lower()
    
    logger.info(f"Downloading {dataset_name} dataset...")
    
    try:
        if dataset_name == 'mnist':
            datasets.MNIST(root=f'{data_root}/raw', train=True, download=True)
            datasets.MNIST(root=f'{data_root}/raw', train=False, download=True)
            
        elif dataset_name == 'svhn':
            datasets.SVHN(root=f'{data_root}/raw', split='train', download=True)
            datasets.SVHN(root=f'{data_root}/raw', split='test', download=True)
            
        elif dataset_name == 'cifar10':
            datasets.CIFAR10(root=f'{data_root}/raw', train=True, download=True)
            datasets.CIFAR10(root=f'{data_root}/raw', train=False, download=True)
            
        elif dataset_name == 'cifar100':
            datasets.CIFAR100(root=f'{data_root}/raw', train=True, download=True)
            datasets.CIFAR100(root=f'{data_root}/raw', train=False, download=True)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        logger.info(f"✓ {dataset_name} dataset downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {dataset_name}: {e}")
        return False
