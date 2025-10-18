import pytest
import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_utils import (
    verify_dataset_integrity,
    get_dataset_stats,
    download_dataset
)


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""
    
    def test_verify_dataset_integrity_invalid_dataset(self):
        """Test that invalid dataset name raises ValueError."""
        with self.assertRaises(ValueError):
            verify_dataset_integrity('invalid_dataset')
    
    @patch('src.data.data_utils.datasets.MNIST')
    def test_verify_dataset_integrity_mnist(self, mock_mnist):
        """Test MNIST dataset verification."""
        # Mock the dataset
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=60000)
        mock_train.__getitem__ = MagicMock(return_value=(MagicMock(), 0))
        
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=10000)
        
        mock_mnist.side_effect = [mock_train, mock_test]
        
        # Test verification
        result = verify_dataset_integrity('mnist', data_root='./test_data')
        self.assertTrue(result)
    
    @patch('src.data.data_utils.datasets.CIFAR10')
    def test_verify_dataset_integrity_cifar10(self, mock_cifar):
        """Test CIFAR-10 dataset verification."""
        # Mock the dataset
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=50000)
        mock_train.__getitem__ = MagicMock(return_value=(MagicMock(), 0))
        
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=10000)
        
        mock_cifar.side_effect = [mock_train, mock_test]
        
        # Test verification
        result = verify_dataset_integrity('cifar10', data_root='./test_data')
        self.assertTrue(result)
    
    @patch('src.data.data_utils.datasets.MNIST')
    def test_verify_dataset_integrity_wrong_size(self, mock_mnist):
        """Test that wrong dataset size returns False."""
        # Mock the dataset with wrong size
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=1000)  # Wrong size
        
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=10000)
        
        mock_mnist.side_effect = [mock_train, mock_test]
        
        # Test verification should fail
        result = verify_dataset_integrity('mnist', data_root='./test_data')
        self.assertFalse(result)
    
    @patch('src.data.data_utils.datasets.MNIST')
    def test_verify_dataset_integrity_file_not_found(self, mock_mnist):
        """Test that FileNotFoundError is handled."""
        mock_mnist.side_effect = FileNotFoundError("Dataset not found")
        
        result = verify_dataset_integrity('mnist', data_root='./test_data')
        self.assertFalse(result)
    
    def test_get_dataset_stats_invalid_dataset(self):
        """Test get_dataset_stats with invalid dataset."""
        stats = get_dataset_stats('invalid_dataset')
        self.assertEqual(stats['train_size'], 0)
        self.assertEqual(stats['test_size'], 0)
    
    @patch('src.data.data_utils.datasets.MNIST')
    def test_get_dataset_stats_mnist(self, mock_mnist):
        """Test getting MNIST dataset stats."""
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=60000)
        
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=10000)
        
        mock_mnist.side_effect = [mock_train, mock_test]
        
        stats = get_dataset_stats('mnist', data_root='./test_data')
        
        self.assertEqual(stats['name'], 'mnist')
        self.assertEqual(stats['train_size'], 60000)
        self.assertEqual(stats['test_size'], 10000)
        self.assertEqual(stats['num_classes'], 10)
        self.assertEqual(stats['image_size'], (28, 28))
        self.assertEqual(stats['num_channels'], 1)
    
    @patch('src.data.data_utils.datasets.CIFAR10')
    def test_get_dataset_stats_cifar10(self, mock_cifar):
        """Test getting CIFAR-10 dataset stats."""
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=50000)
        
        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=10000)
        
        mock_cifar.side_effect = [mock_train, mock_test]
        
        stats = get_dataset_stats('cifar10', data_root='./test_data')
        
        self.assertEqual(stats['name'], 'cifar10')
        self.assertEqual(stats['train_size'], 50000)
        self.assertEqual(stats['test_size'], 10000)
        self.assertEqual(stats['num_classes'], 10)
        self.assertEqual(stats['image_size'], (32, 32))
        self.assertEqual(stats['num_channels'], 3)
    
    @patch('src.data.data_utils.datasets.MNIST')
    def test_download_dataset_mnist(self, mock_mnist):
        """Test downloading MNIST dataset."""
        result = download_dataset('mnist', data_root='./test_data')
        self.assertTrue(result)
        # Verify that MNIST was called twice (train and test)
        self.assertEqual(mock_mnist.call_count, 2)
    
    @patch('src.data.data_utils.datasets.CIFAR10')
    def test_download_dataset_cifar10(self, mock_cifar):
        """Test downloading CIFAR-10 dataset."""
        result = download_dataset('cifar10', data_root='./test_data')
        self.assertTrue(result)
        # Verify that CIFAR10 was called twice (train and test)
        self.assertEqual(mock_cifar.call_count, 2)
    
    def test_download_dataset_invalid(self):
        """Test that invalid dataset name raises ValueError."""
        with self.assertRaises(ValueError):
            download_dataset('invalid_dataset')
    
    def test_dataset_names_case_insensitive(self):
        """Test that dataset names are case insensitive."""
        with patch('src.data.data_utils.datasets.MNIST') as mock_mnist:
            mock_train = MagicMock()
            mock_train.__len__ = MagicMock(return_value=60000)
            mock_train.__getitem__ = MagicMock(return_value=(MagicMock(), 0))
            
            mock_test = MagicMock()
            mock_test.__len__ = MagicMock(return_value=10000)
            
            mock_mnist.side_effect = [mock_train, mock_test]
            
            # Test with different cases
            result = verify_dataset_integrity('MNIST', data_root='./test_data')
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
