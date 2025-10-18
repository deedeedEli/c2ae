#!/bin/bash
set -e

echo "Setting up C2AE reproduction environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/openset_splits
mkdir -p results/checkpoints
mkdir -p results/logs
mkdir -p results/metrics
mkdir -p results/visualizations
mkdir -p results/tensorboard

echo "Environment setup complete!"
echo "Activate with: source venv/bin/activate"
