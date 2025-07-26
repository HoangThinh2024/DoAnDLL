#!/bin/bash

# Advanced start script with uv for Multimodal Pill Recognition System
echo "🚀 Starting Multimodal Pill Recognition System with uv..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Installing uv..."
    pip install uv
    if [ $? -eq 0 ]; then
        print_success "uv installed successfully!"
    else
        print_error "Failed to install uv. Please install manually."
        exit 1
    fi
fi

print_status "uv version: $(uv --version)"

# Create virtual environment with uv
if [ ! -d ".venv" ]; then
    print_status "Creating virtual environment with uv..."
    uv venv
    if [ $? -eq 0 ]; then
        print_success "Virtual environment created!"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv (much faster!)
print_status "Installing dependencies with uv (this is much faster than pip!)..."
uv pip install -e .

if [ $? -eq 0 ]; then
    print_success "Dependencies installed successfully!"
else
    print_warning "Failed to install from pyproject.toml, trying requirements.txt..."
    uv pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed from requirements.txt!"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
fi

# Optional: Install GPU dependencies
read -p "Do you want to install GPU acceleration dependencies (Rapids)? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing GPU dependencies..."
    uv pip install "cudf-cu11>=23.06.0" "cuml-cu11>=23.06.0" "cupy-cuda11x>=12.0.0" || print_warning "GPU dependencies installation failed (normal if no NVIDIA GPU)"
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/raw data/processed checkpoints logs results

# Check system capabilities
print_status "Checking system capabilities..."
python -c "
import torch
import sys
print(f'🐍 Python version: {sys.version}')
print(f'🔥 PyTorch version: {torch.__version__}')
print(f'💻 CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU: {torch.cuda.get_device_name()}')
    print(f'📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('🖥️  Using CPU only')

# Check optional dependencies
try:
    import cudf
    print('✅ Rapids cuDF available')
except ImportError:
    print('⚠️  Rapids cuDF not available')

try:
    import pyspark
    print('✅ Apache Spark available')
except ImportError:
    print('⚠️  Apache Spark not available')
"

# Ask user what to run
echo
echo "What would you like to do?"
echo "1) Run Streamlit app"
echo "2) Run Jupyter notebook"
echo "3) Run training script"
echo "4) Run data processing"
echo "5) Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        print_status "Starting Streamlit application..."
        print_success "🌐 Open your browser and go to: http://localhost:8501"
        streamlit run app.py
        ;;
    2)
        print_status "Starting Jupyter Lab..."
        uv pip install jupyterlab
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
        ;;
    3)
        print_status "Running training script..."
        python src/training/trainer.py
        ;;
    4)
        print_status "Running data processing..."
        python -c "
from src.data.data_processing import SparkDataProcessor
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

processor = SparkDataProcessor(config)
processor.create_sample_dataset('data/raw/sample.parquet', 1000)
print('✅ Sample data created!')
"
        ;;
    5)
        print_success "Goodbye!"
        exit 0
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

print_success "✅ Process completed!"
