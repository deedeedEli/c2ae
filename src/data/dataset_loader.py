import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from typing import List, Dict, Tuple, Optional
import numpy as np
from .transforms import get_train_transforms, get_test_transforms


class OpenSetDataset(Dataset):
    """
    PyTorch Dataset for open-set recognition.
    
    Args:
        dataset_name: 'mnist', 'svhn', 'cifar10', 'cifar+10', 'cifar+50', 'tiny_imagenet'
        known_classes: List of known class indices
        split: 'train', 'test_known', 'test_unknown'
        include_background: Whether to include background class in training
        transform: Data transformation pipeline
        data_root: Root directory for datasets
    """
    
    def __init__(self, 
                 dataset_name: str, 
                 known_classes: List[int], 
                 split: str = 'train',
                 include_background: bool = False,
                 transform = None,
                 data_root: str = './data'):
        
        self.dataset_name = dataset_name.lower()
        self.known_classes = set(known_classes)
        self.split = split
        self.include_background = include_background
        self.transform = transform
        self.data_root = data_root
        
        # Load base dataset
        self.base_dataset, self.num_total_classes = self._load_base_dataset()
        
        # Filter data based on split
        self.data, self.labels, self.is_known = self._filter_data()
        
    def _load_base_dataset(self):
        """Load the base dataset."""
        is_train = (self.split == 'train')
        
        if self.dataset_name == 'mnist':
            dataset = datasets.MNIST(
                root=f'{self.data_root}/raw',
                train=is_train,
                download=True,
                transform=None
            )
            num_classes = 10
            
        elif self.dataset_name == 'svhn':
            split_name = 'train' if is_train else 'test'
            dataset = datasets.SVHN(
                root=f'{self.data_root}/raw',
                split=split_name,
                download=True,
                transform=None
            )
            num_classes = 10
            
        elif self.dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(
                root=f'{self.data_root}/raw',
                train=is_train,
                download=True,
                transform=None
            )
            num_classes = 10
            
        elif self.dataset_name == 'cifar100':
            dataset = datasets.CIFAR100(
                root=f'{self.data_root}/raw',
                train=is_train,
                download=True,
                transform=None
            )
            num_classes = 100
            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return dataset, num_classes
    
    def _filter_data(self) -> Tuple[List, List, List]:
        """Filter data based on known/unknown split."""
        data = []
        labels = []
        is_known = []
        
        # Get all data and labels from base dataset
        if self.dataset_name in ['mnist', 'cifar10', 'cifar100']:
            all_data = self.base_dataset.data
            all_labels = np.array(self.base_dataset.targets)
        elif self.dataset_name == 'svhn':
            all_data = self.base_dataset.data.transpose(0, 2, 3, 1)  # SVHN is (N, C, H, W), convert to (N, H, W, C)
            all_labels = self.base_dataset.labels
        
        for i in range(len(all_labels)):
            label = int(all_labels[i])
            
            if self.split == 'train':
                # Training: include only known classes
                if label in self.known_classes:
                    data.append(all_data[i])
                    labels.append(label)
                    is_known.append(True)
                    
            elif self.split == 'test_known':
                # Test known: only known classes
                if label in self.known_classes:
                    data.append(all_data[i])
                    labels.append(label)
                    is_known.append(True)
                    
            elif self.split == 'test_unknown':
                # Test unknown: only unknown classes
                if label not in self.known_classes:
                    data.append(all_data[i])
                    labels.append(label)
                    is_known.append(False)
        
        return data, labels, is_known
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label, is_known)
                - image: torch.Tensor, shape (C, H, W)
                - label: int, original class label
                - is_known: bool, whether sample is from known classes
        """
        from PIL import Image
        
        img = self.data[idx]
        label = self.labels[idx]
        is_known_flag = self.is_known[idx]
        
        # Convert to PIL Image
        if isinstance(img, np.ndarray):
            if img.ndim == 2:  # Grayscale
                img = Image.fromarray(img, mode='L')
            else:  # RGB
                img = Image.fromarray(img, mode='RGB')
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, is_known_flag


def get_openset_loaders(dataset_name: str, 
                       known_classes: List[int], 
                       batch_size: int,
                       config: Dict,
                       num_workers: int = 4, 
                       data_root: str = './data') -> Dict[str, DataLoader]:
    """
    Create train and test data loaders for open-set experiments.
    
    Args:
        dataset_name: Dataset name
        known_classes: Known class indices
        batch_size: Batch size
        config: Configuration dictionary with augmentation settings
        num_workers: Number of data loading workers
        data_root: Data directory
    
    Returns:
        dict: {
            'train': DataLoader for training (known classes)
            'test_known': DataLoader for known class testing
            'test_unknown': DataLoader for unknown class testing
        }
    """
    # Get image properties
    if dataset_name.lower() == 'mnist':
        image_size = 28
        image_channels = 1
    elif dataset_name.lower() in ['svhn', 'cifar10', 'cifar100']:
        image_size = 32
        image_channels = 3
    else:
        image_size = 32
        image_channels = 3
    
    # Get transforms
    train_transform = get_train_transforms(
        config.get('augmentation', {}).get('train', {}),
        image_size,
        image_channels
    )
    test_transform = get_test_transforms(
        config.get('augmentation', {}).get('test', {}),
        image_size,
        image_channels
    )
    
    # Create datasets
    train_dataset = OpenSetDataset(
        dataset_name=dataset_name,
        known_classes=known_classes,
        split='train',
        include_background=False,
        transform=train_transform,
        data_root=data_root
    )
    
    test_known_dataset = OpenSetDataset(
        dataset_name=dataset_name,
        known_classes=known_classes,
        split='test_known',
        include_background=False,
        transform=test_transform,
        data_root=data_root
    )
    
    test_unknown_dataset = OpenSetDataset(
        dataset_name=dataset_name,
        known_classes=known_classes,
        split='test_unknown',
        include_background=False,
        transform=test_transform,
        data_root=data_root
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_known_loader = DataLoader(
        test_known_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_unknown_loader = DataLoader(
        test_unknown_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test_known': test_known_loader,
        'test_unknown': test_unknown_loader
    }
