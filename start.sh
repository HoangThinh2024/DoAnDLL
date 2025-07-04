#!/bin/bash

# Quick start script for Multimodal Pill Recognition System
echo "🚀 Starting Multimodal Pill Recognition System..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    uv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv (much faster than pip!)
echo "⚡ Installing dependencies with uv..."
uv pip install -r requirements-minimal.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed checkpoints logs results

# Check if CUDA is available
python -c "import torch; print('🔥 CUDA available:', torch.cuda.is_available())"

# Run Streamlit app
echo "🌐 Starting Streamlit application..."
echo "📍 Open your browser and go to: http://localhost:8501"
streamlit run app.py

echo "✅ Application started successfully!"
