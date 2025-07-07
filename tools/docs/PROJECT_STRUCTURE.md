# 📁 Project Structure

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
