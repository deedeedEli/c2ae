#!/bin/bash
set -e

echo "Downloading datasets for C2AE..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Create data directory
mkdir -p data/raw

# Download MNIST
echo "Downloading MNIST..."
python -c "
from torchvision import datasets
datasets.MNIST(root='./data/raw', train=True, download=True)
datasets.MNIST(root='./data/raw', train=False, download=True)
print('MNIST downloaded successfully!')
"

# Download CIFAR-10
echo "Downloading CIFAR-10..."
python -c "
from torchvision import datasets
datasets.CIFAR10(root='./data/raw', train=True, download=True)
datasets.CIFAR10(root='./data/raw', train=False, download=True)
print('CIFAR-10 downloaded successfully!')
"

# Download SVHN
echo "Downloading SVHN..."
python -c "
from torchvision import datasets
datasets.SVHN(root='./data/raw', split='train', download=True)
datasets.SVHN(root='./data/raw', split='test', download=True)
print('SVHN downloaded successfully!')
"

echo "All datasets downloaded successfully!"
