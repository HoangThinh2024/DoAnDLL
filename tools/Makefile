# Smart Pill Recognition System Makefile

.PHONY: help setup clean test train web cli docker

# Default target
help:
	@echo "🔥 Smart Pill Recognition System"
	@echo "Available targets:"
	@echo "  setup     - Setup environment and install dependencies"
	@echo "  clean     - Clean cache files and temporary files" 
	@echo "  test      - Run tests"
	@echo "  train     - Train the model"
	@echo "  web       - Launch web UI"
	@echo "  cli       - Launch CLI"
	@echo "  docker    - Build and run Docker container"
	@echo "  status    - Show system status"

# Setup environment
setup:
	@echo "🔧 Setting up environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	python main.py setup

# Clean cache files
clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

# Train model
train:
	@echo "🏋️ Training model..."
	python main.py train

# Launch web UI
web:
	@echo "🌐 Launching web UI..."
	python main.py web

# Launch CLI
cli:
	@echo "🖥️ Launching CLI..."
	python main.py cli

# Build and run Docker
docker:
	@echo "🐳 Building Docker image..."
	docker-compose up --build

# Show status
status:
	@echo "📊 System status..."
	python main.py status

# Quick recognition test
demo:
	@echo "🎯 Running demo recognition..."
	@if [ -f "Dataset_BigData/CURE_dataset/CURE_dataset_test/0_bottom_24.jpg" ]; then \
		python main.py recognize Dataset_BigData/CURE_dataset/CURE_dataset_test/0_bottom_24.jpg; \
	else \
		echo "❌ Demo image not found"; \
	fi

# Install in development mode
install-dev:
	@echo "👨‍💻 Installing in development mode..."
	pip install -e .

# Format code
format:
	@echo "✨ Formatting code..."
	python -m black . --line-length 100
	python -m isort . --profile black
