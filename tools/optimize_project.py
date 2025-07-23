#!/usr/bin/env python3
"""
🧹 Project Optimizer & Cleaner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Script tối ưu hóa và dọn dẹp project để có cấu trúc gọn gàng và hiệu quả

Tác giả: DoAnDLL Project
Ngày: 2025
"""

import os
import shutil
import sys
from pathlib import Path
import json
import yaml
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent

class ProjectOptimizer:
    """Lớp tối ưu hóa project"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.moved_files = []
        self.cleaned_files = []
        
    def show_banner(self):
        """Hiển thị banner"""
        print("""
🧹 PROJECT OPTIMIZER & CLEANER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tối ưu hóa cấu trúc project cho Smart Pill Recognition System
        """)
    
    def organize_files(self):
        """Tổ chức lại files vào các thư mục phù hợp"""
        print("📁 Đang tổ chức lại files...")
        
        # File mapping - source: destination
        file_moves = {
            # Move training scripts to scripts/
            "train_cure_model.py": "scripts/train_cure_model.py",
            "train_optimized_server.py": "scripts/train_optimized_server.py",
            
            # Move recognition script to apps/cli/
            "recognize.py": "apps/cli/recognize.py",
            
            # Move test scripts to tools/
            "quick_test.py": "tools/quick_test.py",
            "test_dataset_port": "tools/test_dataset_port",
            
            # Move standalone apps to apps/
            "app_with_dataset.py": "apps/legacy/app_with_dataset.py",
        }
        
        # Create directories if they don't exist
        dirs_to_create = [
            "scripts",
            "tools", 
            "apps/legacy",
            "docs"
        ]
        
        for dir_path in dirs_to_create:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  📂 Created: {dir_path}/")
        
        # Move files
        for source, destination in file_moves.items():
            source_path = self.project_root / source
            dest_path = self.project_root / destination
            
            if source_path.exists():
                # Create destination directory if needed
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(source_path), str(dest_path))
                self.moved_files.append((source, destination))
                print(f"  📦 Moved: {source} → {destination}")
        
        print(f"✅ Đã tổ chức {len(self.moved_files)} files")
    
    def clean_cache_files(self):
        """Dọn dẹp cache files và temporary files"""
        print("🧹 Đang dọn dẹp cache files...")
        
        # Patterns to clean
        patterns_to_clean = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo", 
            "**/.pytest_cache",
            "**/.coverage",
            "**/logs/*.log",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.temp"
        ]
        
        cleaned_count = 0
        
        for pattern in patterns_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    path.unlink()
                    cleaned_count += 1
                    print(f"  🗑️  Removed: {path.relative_to(self.project_root)}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    cleaned_count += 1
                    print(f"  🗑️  Removed: {path.relative_to(self.project_root)}/")
        
        print(f"✅ Đã dọn dẹp {cleaned_count} files/folders")
    
    def create_project_structure_doc(self):
        """Tạo documentation về cấu trúc project"""
        print("📝 Đang tạo project structure documentation...")
        
        structure_doc = """# 📁 Project Structure

```
Smart Pill Recognition System/
├── 🚀 main.py                     # Main launcher script
├── 📋 requirements.txt            # Python dependencies  
├── ⚙️ pyproject.toml              # Project configuration
├── 🐳 Dockerfile                  # Docker configuration
├── 🐳 docker-compose.yml          # Docker Compose
├── 📖 README.md                   # Project documentation
├── 📄 LICENSE                     # License file
├── 🔧 Makefile                    # Build automation
│
├── 📱 apps/                       # Applications
│   ├── 🖥️ cli/                   # Command Line Interface
│   │   ├── main.py               # CLI main script
│   │   └── recognize.py          # CLI recognition tool
│   ├── 🌐 web/                   # Web Interface  
│   │   └── streamlit_app.py      # Streamlit web app
│   └── 📚 legacy/                # Legacy applications
│       └── app_with_dataset.py   # Old app version
│
├── 🧠 core/                      # Core modules
│   ├── 📊 data/                  # Data processing
│   │   ├── __init__.py
│   │   ├── cure_dataset.py       # CURE dataset handler
│   │   └── data_processing.py    # Data preprocessing
│   ├── 🤖 models/                # AI Models
│   │   ├── __init__.py
│   │   └── multimodal_transformer.py
│   ├── 🏋️ training/              # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── 🔧 utils/                 # Utility functions
│       ├── __init__.py
│       ├── metrics.py
│       ├── port_manager.py
│       └── utils.py
│
├── 📜 scripts/                   # Training & utility scripts
│   ├── train_cure_model.py       # Main training script
│   └── train_optimized_server.py # Optimized training
│
├── 🛠️ tools/                    # Development tools
│   ├── quick_test.py             # Quick testing
│   └── test_dataset_port         # Dataset port testing
│
├── ⚙️ config/                    # Configuration files
│   └── config.yaml               # Main configuration
│
├── 💾 checkpoints/               # Model checkpoints
│   └── (model files...)
│
├── 📊 data/                      # Processed data
│   ├── processed/                # Processed datasets
│   └── raw/                      # Raw datasets
│
├── 📈 Dataset_BigData/           # Big datasets
│   └── CURE_dataset/             # CURE dataset
│       ├── recognition.py
│       ├── train.py
│       ├── u2net.py
│       ├── CURE_dataset_test/
│       ├── CURE_dataset_train_cut_bounding_box/
│       ├── CURE_dataset_validation_cut_bounding_box/
│       └── fonts/
│
├── 📓 notebooks/                 # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── demo_multimodal_pill_recognition.ipynb
│   ├── model_experiments.ipynb
│   └── results_analysis.ipynb
│
├── 📝 logs/                      # Log files
│   └── (log files...)
│
└── 📚 docs/                      # Documentation
    └── (documentation files...)
```

## 🎯 Key Components

### 🚀 Main Launcher (`main.py`)
- Central entry point for all operations
- Supports both CLI and Web UI modes
- Handles setup, training, and recognition

### 📱 Applications (`apps/`)
- **CLI**: Rich terminal interface with beautiful formatting
- **Web**: Streamlit-based web interface with modern UI
- **Legacy**: Older versions for compatibility

### 🧠 Core Modules (`core/`)
- **Data**: Dataset handling and preprocessing
- **Models**: AI model implementations  
- **Training**: Training loops and utilities
- **Utils**: Common utility functions

### 📜 Scripts (`scripts/`)
- Training scripts for different scenarios
- Optimization and performance scripts

### 🛠️ Tools (`tools/`)
- Development and debugging tools
- Testing utilities
- Performance benchmarks

## 🚀 Quick Start

```bash
# Setup environment
python main.py setup

# Launch CLI
python main.py cli

# Launch Web UI  
python main.py web

# Quick recognition
python main.py recognize image.jpg

# Train model
python main.py train
```

## 📖 More Information

- See `README.md` for detailed setup instructions
- Check `notebooks/` for examples and tutorials
- Visit `docs/` for comprehensive documentation
"""
        
        docs_dir = self.project_root / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        structure_file = docs_dir / "PROJECT_STRUCTURE.md"
        structure_file.write_text(structure_doc)
        
        print(f"  📝 Created: {structure_file.relative_to(self.project_root)}")
    
    def update_import_paths(self):
        """Cập nhật import paths trong các files đã move"""
        print("🔄 Đang cập nhật import paths...")
        
        # Files to update and their new import paths
        files_to_update = [
            ("apps/cli/recognize.py", "../../core"),
            ("apps/web/streamlit_app.py", "../../core"),
            ("scripts/train_cure_model.py", "../core")
        ]
        
        for file_path, new_import_base in files_to_update:
            full_path = self.project_root / file_path
            
            if full_path.exists():
                # Read file content
                content = full_path.read_text()
                
                # Update sys.path.append statements
                old_patterns = [
                    "sys.path.append('src')",
                    "sys.path.append(str(Path(__file__).parent))",
                    "sys.path.append('.')"
                ]
                
                for old_pattern in old_patterns:
                    if old_pattern in content:
                        content = content.replace(
                            old_pattern,
                            f"sys.path.append('{new_import_base}')"
                        )
                
                # Write back
                full_path.write_text(content)
                print(f"  🔄 Updated imports: {file_path}")
    
    def create_makefile(self):
        """Tạo Makefile cho automation"""
        print("⚙️ Đang tạo Makefile...")
        
        makefile_content = """# Smart Pill Recognition System Makefile

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
	@if [ -f "Dataset_BigData/CURE_dataset/CURE_dataset_test/0_bottom_24.jpg" ]; then \\
		python main.py recognize Dataset_BigData/CURE_dataset/CURE_dataset_test/0_bottom_24.jpg; \\
	else \\
		echo "❌ Demo image not found"; \\
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
"""
        
        makefile_path = self.project_root / "Makefile"
        makefile_path.write_text(makefile_content)
        
        print(f"  ⚙️ Created: {makefile_path.relative_to(self.project_root)}")
    
    def create_run_script(self):
        """Tạo script chạy nhanh"""
        print("🚀 Đang tạo run script...")
        
        run_script_content = """#!/bin/bash
# Smart Pill Recognition System - Quick Run Script

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
PURPLE='\\033[0;35m'
CYAN='\\033[0;36m'
NC='\\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "   ██████╗ ██╗██╗     ██╗         ██████╗ ███████╗ ██████╗ ███████╗"
echo "   ██╔══██╗██║██║     ██║         ██╔══██╗██╔════╝██╔═══██╗██╔════╝"
echo "   ██████╔╝██║██║     ██║         ██████╔╝█████╗  ██║   ██║█████╗  "
echo "   ██╔═══╝ ██║██║     ██║         ██╔══██╗██╔══╝  ██║   ██║██╔══╝  "
echo "   ██║     ██║███████╗███████╗    ██║  ██║███████╗╚██████╔╝███████╗"
echo "   ╚═╝     ╚═╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝"
echo -e "${NC}"
echo -e "${YELLOW}              🔥 SMART PILL RECOGNITION SYSTEM 🔥${NC}"
echo -e "${CYAN}                AI-Powered Pharmaceutical Identification Platform${NC}"
echo ""

# Function to show help
show_help() {
    echo -e "${GREEN}Usage: ./run [COMMAND]${NC}"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo -e "  ${YELLOW}setup${NC}     - Setup environment and install dependencies"
    echo -e "  ${YELLOW}cli${NC}       - Launch CLI interface"
    echo -e "  ${YELLOW}web${NC}       - Launch web interface"
    echo -e "  ${YELLOW}train${NC}     - Train the model"
    echo -e "  ${YELLOW}test${NC}      - Run tests"
    echo -e "  ${YELLOW}clean${NC}     - Clean cache files"
    echo -e "  ${YELLOW}status${NC}    - Show system status"
    echo -e "  ${YELLOW}help${NC}      - Show this help message"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo -e "  ./run setup                    # First time setup"
    echo -e "  ./run cli                      # Launch CLI"
    echo -e "  ./run web                      # Launch web UI"
    echo ""
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed!${NC}"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    min_version="3.10"
    
    if [ "$(printf '%s\\n' "$min_version" "$python_version" | sort -V | head -n1)" = "$min_version" ]; then
        echo -e "${GREEN}✅ Python $python_version detected${NC}"
    else
        echo -e "${RED}❌ Python $min_version+ required, but $python_version found${NC}"
        exit 1
    fi
}

# Main logic
main() {
    check_python
    
    case "${1:-help}" in
        setup)
            echo -e "${CYAN}🔧 Setting up environment...${NC}"
            python3 main.py setup
            ;;
        cli)
            echo -e "${CYAN}🖥️ Launching CLI...${NC}"
            python3 main.py cli
            ;;
        web)
            echo -e "${CYAN}🌐 Launching Web UI...${NC}"
            echo -e "${GREEN}🔗 Will open at: http://localhost:8501${NC}"
            python3 main.py web
            ;;
        train)
            echo -e "${CYAN}🏋️ Starting training...${NC}"
            python3 main.py train
            ;;
        test)
            echo -e "${CYAN}🧪 Running tests...${NC}"
            if [ -f "tests/test_main.py" ]; then
                python3 -m pytest tests/ -v
            else
                echo -e "${YELLOW}⚠️ No tests found${NC}"
            fi
            ;;
        clean)
            echo -e "${CYAN}🧹 Cleaning cache files...${NC}"
            find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
            find . -type f -name "*.pyc" -delete 2>/dev/null || true
            echo -e "${GREEN}✅ Cleanup completed${NC}"
            ;;
        status)
            echo -e "${CYAN}📊 Checking system status...${NC}"
            python3 main.py status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}❌ Unknown command: $1${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
"""
        
        run_script_path = self.project_root / "run"
        run_script_path.write_text(run_script_content)
        
        # Make executable
        os.chmod(run_script_path, 0o755)
        
        print(f"  🚀 Created: {run_script_path.relative_to(self.project_root)}")
    
    def optimize_requirements(self):
        """Tối ưu hóa requirements.txt"""
        print("📦 Đang tối ưu hóa requirements.txt...")
        
        # Read current requirements
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            print("  ❌ requirements.txt không tồn tại")
            return
        
        # Group requirements by category and optimize
        optimized_requirements = """# Smart Pill Recognition System - Requirements
# Tối ưu hóa cho Ubuntu 22.04 + NVIDIA Quadro 6000 + CUDA 12.8

# ================================
# 🧠 Core AI/ML Dependencies
# ================================
torch>=2.3.0
torchvision>=0.18.0
torchaudio>=2.3.0
transformers>=4.40.0
timm>=1.0.3

# ================================
# 📊 Data Science & Processing 
# ================================
numpy>=1.26.0
pandas>=2.2.0
Pillow>=10.2.0
opencv-python-headless>=4.9.0
scikit-learn>=1.4.0

# ================================
# 🌐 Web UI & Visualization
# ================================
streamlit>=1.25.0
streamlit-option-menu>=0.3.6
plotly>=5.15.0
matplotlib>=3.8.0
seaborn>=0.13.0

# ================================
# 🔧 CLI & Terminal UI
# ================================
rich>=13.7.0
typer>=0.9.0
click>=8.1.0

# ================================
# 📈 Big Data & Performance
# ================================
pyspark>=3.4.0
findspark>=2.0.1
pyarrow>=12.0.0

# ================================
# 🔍 Search & Indexing
# ================================
elasticsearch>=8.8.0
faiss-cpu>=1.7.4

# ================================
# ☁️ Cloud & API
# ================================
requests>=2.31.0
fastapi>=0.104.0
uvicorn>=0.24.0

# ================================
# 🛠️ Development & Testing
# ================================
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.12.0
isort>=5.13.0
flake8>=6.1.0

# ================================
# 📝 Documentation & Utilities  
# ================================
pyyaml>=6.0.1
tqdm>=4.66.0
psutil>=5.9.6
python-dotenv>=1.0.0
"""
        
        # Write optimized requirements
        req_file.write_text(optimized_requirements)
        print(f"  📦 Optimized: {req_file.relative_to(self.project_root)}")
    
    def create_gitignore(self):
        """Tạo .gitignore file"""
        print("📝 Đang tạo .gitignore...")
        
        gitignore_content = """# Smart Pill Recognition System - .gitignore

# ================================
# 🐍 Python
# ================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# ================================
# 🧠 Machine Learning
# ================================
*.pth
*.pt
*.h5
*.hdf5
*.pkl
*.pickle
*.joblib
*.model
checkpoints/
models/weights/
wandb/
mlruns/

# ================================
# 📊 Data Files
# ================================
*.csv
*.tsv
*.json
*.jsonl
*.parquet
*.feather
data/raw/
data/processed/
Dataset_BigData/CURE_dataset_train_cut_bounding_box/
Dataset_BigData/CURE_dataset_validation_cut_bounding_box/
!Dataset_BigData/CURE_dataset/CURE_dataset_test/

# ================================
# 📝 Logs & Output
# ================================
*.log
logs/
outputs/
results/
experiments/
*.out
*.err

# ================================
# 🔧 IDEs & Editors
# ================================
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# ================================
# 🐳 Docker
# ================================
.dockerignore
docker-compose.override.yml

# ================================
# ☁️ Cloud & Deployment
# ================================
.env
.env.local
.env.production
*.key
*.pem
.aws/
.gcp/
.azure/

# ================================
# 📦 Package Managers
# ================================
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# ================================
# 🧪 Testing
# ================================
.coverage
.pytest_cache/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# ================================
# 📱 Jupyter Notebooks
# ================================
.ipynb_checkpoints
*.ipynb

# ================================
# 🔒 Secrets
# ================================
secrets.yaml
config/local.yaml
.secret
*.secret

# ================================
# 📊 Large Files
# ================================
*.zip
*.tar.gz
*.7z
*.rar
"""
        
        gitignore_path = self.project_root / ".gitignore"
        gitignore_path.write_text(gitignore_content)
        
        print(f"  📝 Created: {gitignore_path.relative_to(self.project_root)}")
    
    def show_summary(self):
        """Hiển thị tóm tắt kết quả"""
        print("\n" + "=" * 80)
        print("🎉 TỐI ƯU HÓA PROJECT HOÀN THÀNH!")
        print("=" * 80)
        
        print(f"\n📦 Files đã di chuyển: {len(self.moved_files)}")
        for source, dest in self.moved_files:
            print(f"  📁 {source} → {dest}")
        
        print(f"\n🧹 Files đã dọn dẹp: {len(self.cleaned_files)}")
        
        print("\n🚀 Cấu trúc project mới:")
        print("""
📁 Smart Pill Recognition System/
├── 🚀 main.py                   # Main launcher
├── 🏃 run                       # Quick run script  
├── ⚙️ Makefile                  # Build automation
├── 📋 requirements.txt          # Dependencies
├── 🙈 .gitignore               # Git ignore rules
│
├── 📱 apps/                     # Applications
│   ├── 🖥️ cli/                 # CLI interface
│   ├── 🌐 web/                 # Web interface
│   └── 📚 legacy/              # Legacy apps
│
├── 🧠 core/                    # Core modules
│   ├── 📊 data/                # Data processing
│   ├── 🤖 models/              # AI models
│   ├── 🏋️ training/            # Training
│   └── 🔧 utils/               # Utilities
│
├── 📜 scripts/                 # Scripts
├── 🛠️ tools/                  # Dev tools
├── 📚 docs/                    # Documentation
└── ...
        """)
        
        print("\n🎯 Cách sử dụng mới:")
        print("  ./run setup      # Setup environment")
        print("  ./run cli        # Launch CLI")  
        print("  ./run web        # Launch Web UI")
        print("  make help        # See all commands")
        print("  python main.py   # Direct launcher")
        
        print("\n✨ Project đã được tối ưu hóa và sẵn sàng sử dụng!")
    
    def run_optimization(self):
        """Chạy toàn bộ quá trình tối ưu hóa"""
        self.show_banner()
        
        try:
            self.organize_files()
            self.clean_cache_files()
            self.update_import_paths()
            self.create_project_structure_doc()
            self.create_makefile()
            self.create_run_script()
            self.optimize_requirements()
            self.create_gitignore()
            self.show_summary()
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình tối ưu hóa: {e}")
            return False
        
        return True

def main():
    """Main function"""
    optimizer = ProjectOptimizer()
    
    if optimizer.run_optimization():
        print("\n🎉 Tối ưu hóa thành công!")
        return 0
    else:
        print("\n❌ Tối ưu hóa thất bại!")
        return 1

if __name__ == "__main__":
    exit(main())
