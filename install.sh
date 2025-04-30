#!/bin/bash

echo "Installing Genesis OS..."

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed! Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv || {
    echo "Failed to create virtual environment!"
    exit 1
}

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || {
    echo "Failed to activate virtual environment!"
    exit 1
}

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -e .
else
    echo "No NVIDIA GPU detected. Installing CPU version..."
    pip install -e .
fi

echo
echo "Installation complete! You can now run Genesis OS using:"
echo "python run_genesis.py"
echo

# Make the script executable
chmod +x run_genesis.py 