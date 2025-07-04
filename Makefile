# Makefile for Multimodal Pill Recognition System

.PHONY: help install install-gpu run train notebook clean test lint format docker-build docker-run

# Default target
help:
	@echo "🚀 Multimodal Pill Recognition System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make install-gpu  - Install with GPU support"
	@echo "  make run          - Run Streamlit app"
	@echo "  make train        - Run training script"
	@echo "  make notebook     - Start Jupyter Lab"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean cache and temp files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run with Docker"

# Setup virtual environment and install dependencies
install:
	@echo "📦 Setting up environment with uv..."
	uv venv --python 3.8
	uv pip install -e .
	@echo "✅ Installation complete!"

# Install with GPU support
install-gpu: install
	@echo "🎮 Installing GPU dependencies..."
	uv pip install "cudf-cu11>=23.06.0" "cuml-cu11>=23.06.0" "cupy-cuda11x>=12.0.0" || echo "⚠️ GPU dependencies failed (normal without NVIDIA GPU)"

# Run Streamlit app
run:
	@echo "🌐 Starting Streamlit app..."
	@echo "Open your browser at: http://localhost:8501"
	streamlit run app.py

# Run training
train:
	@echo "🏋️ Starting model training..."
	python src/training/trainer.py

# Start Jupyter Lab
notebook:
	@echo "📓 Starting Jupyter Lab..."
	uv pip install jupyterlab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Run tests
test:
	@echo "🧪 Running tests..."
	uv pip install pytest pytest-cov
	pytest tests/ -v --cov=src

# Lint code
lint:
	@echo "🔍 Running linting..."
	uv pip install flake8 mypy
	flake8 src/ app.py
	mypy src/ --ignore-missing-imports

# Format code
format:
	@echo "✨ Formatting code..."
	uv pip install black isort
	black src/ app.py
	isort src/ app.py

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/

# Build Docker image
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t pill-recognition .

# Run with Docker
docker-run:
	@echo "🚀 Running with Docker..."
	docker run -p 8501:8501 -v $(PWD)/data:/app/data pill-recognition

# Full setup (alternative to scripts)
setup: install
	@echo "📁 Creating directories..."
	mkdir -p data/raw data/processed checkpoints logs results
	@echo "🎉 Setup complete! Run 'make run' to start the app."

# Development setup
dev-setup: install
	@echo "🛠️ Setting up development environment..."
	uv pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Development setup complete!"

# Check system
check:
	@echo "🔍 Checking system capabilities..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
	@uv --version
	@echo "✅ System check complete!"
