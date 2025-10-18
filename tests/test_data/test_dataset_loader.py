import torch
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset_loader import OpenSetDataset, get_openset_loaders
from src.utils.config import Config


@pytest.mark.parametrize("dataset_name", ["mnist", "cifar10"])
def test_openset_dataset_train_split(dataset_name):
    """Test OpenSetDataset train split."""
    known_classes = [0, 1, 2, 3, 4, 5]
    
    dataset = OpenSetDataset(
        dataset_name=dataset_name,
        known_classes=known_classes,
        split='train',
        data_root='./data'
    )
    
    # Check that we have samples
    assert len(dataset) > 0
    
    # Check that all samples are from known classes
    for i in range(min(10, len(dataset))):
        _, label, is_known = dataset[i]
        assert is_known == True
        assert label in known_classes


@pytest.mark.parametrize("dataset_name", ["mnist", "cifar10"])
def test_openset_dataset_test_unknown_split(dataset_name):
    """Test OpenSetDataset test unknown split."""
    known_classes = [0, 1, 2, 3, 4, 5]
    
    dataset = OpenSetDataset(
        dataset_name=dataset_name,
        known_classes=known_classes,
        split='test_unknown',
        data_root='./data'
    )
    
    # Check that we have samples
    assert len(dataset) > 0
    
    # Check that all samples are from unknown classes
    for i in range(min(10, len(dataset))):
        _, label, is_known = dataset[i]
        assert is_known == False
        assert label not in known_classes


def test_get_openset_loaders():
    """Test get_openset_loaders function."""
    config = Config()
    known_classes = [0, 1, 2, 3, 4, 5]
    
    loaders = get_openset_loaders(
        dataset_name='mnist',
        known_classes=known_classes,
        batch_size=32,
        config=config.config,
        num_workers=0,
        data_root='./data'
    )
    
    assert 'train' in loaders
    assert 'test_known' in loaders
    assert 'test_unknown' in loaders
    
    # Check data loader works
    batch = next(iter(loaders['train']))
    images, labels, is_known = batch
    
    assert images.shape[0] <= 32
    assert labels.shape[0] == images.shape[0]
    assert is_known.shape[0] == images.shape[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
