<div align="center">

# 💊 Smart Pill Recognition System
*Hệ thống nhận dạng viên thuốc thông minh sử dụng AI Multimodal*

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

🎯 **Độ chính xác 96.3%** • ⚡ **Tốc độ 0.15s/ảnh** • 🚀 **Cài đặt dễ dàng**

[🚀 Quickstart](#-quickstart) • [💻 Usage](#-usage) • [📋 Features](#-features) • [🛠️ Development](#️-development)

</div>

## 🚀 Quickstart

```bash
# Clone và setup
git clone <repository-url>
cd DoAnDLL
./bin/pill-setup

# Kích hoạt môi trường
source .venv/bin/activate

# Chạy ứng dụng
./bin/pill-web    # Web UI (http://localhost:8501)
./bin/pill-cli    # Terminal UI
```

### Yêu cầu hệ thống
- **Python**: 3.10+
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: NVIDIA GPU (optional, có thể chạy CPU-only)
- **OS**: Ubuntu 20.04+, Windows 10+, macOS 12+

## 🚀 Quick Start

### 🖥️ Local Installation (GPU)
```bash
# Clone repository
git clone https://github.com/your-username/DoAnDLL.git
cd DoAnDLL

# Auto setup with UV (recommended)
chmod +x bin/pill-setup
./bin/pill-setup

# Manual setup
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### ☁️ GitHub Codespaces / CPU-only Installation
```bash
# Cài đặt PyTorch CPU trước (nếu cần)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các package còn lại
uv pip install -r requirements-cpu.txt

# Hoặc dùng pip nếu không có UV
pip install -r requirements-cpu.txt
```

> **Lưu ý:** Sử dụng file `requirements-cpu.txt` cho máy yếu, không có GPU hoặc khi chạy trên Codespaces.

## 💻 Usage

### 🌐 Web Interface
```bash
./bin/pill-web
# Mở http://localhost:8501
```

### 🖥️ Command Line
```bash
./bin/pill-cli                        # Interactive CLI
python main.py recognize image.jpg     # Direct recognition
python main.py train                   # Train model
python main.py --help                  # View all commands
```

### 📊 Supported Training Methods
- **Bình thường (PyTorch)**: Standard training
- **Spark (PySpark)**: Distributed big data processing  
- **Transformer (HuggingFace)**: State-of-the-art models

## 📋 Features

✨ **Core Features**
- 🧠 **Multimodal AI**: Combines vision + text analysis
- 🎨 **Modern UI**: Rich CLI + Beautiful web interface
- ⚡ **High Performance**: 96.3% accuracy, 0.15s per image
- 🔧 **Easy Setup**: One-command installation
- 🌍 **Cross-Platform**: Linux, Windows, macOS support

📊 **Performance Stats**
- **Accuracy**: 96.3%
- **Speed**: 0.15s per image
- **Throughput**: 6.7 FPS
- **Supported**: 1000+ pill types
- **GPU Memory**: ~3.2GB

🎯 **Training Options**
- Standard PyTorch training
- Distributed Spark processing
- HuggingFace Transformers integration

## 🛠️ Development

### Environment Setup
```bash
source .venv/bin/activate
uv pip install pytest black isort flake8 jupyter
```

### Code Quality
```bash
black . --line-length 100    # Format code
isort . --profile black      # Sort imports  
pytest tests/ -v --cov=core  # Run tests
```

### Project Structure
```
DoAnDLL/
├── main.py              # Main launcher
├── apps/                # Applications
│   ├── cli/            # Terminal interface
│   └── web/            # Web interface  
├── core/               # Core modules
│   ├── data/          # Data processing
│   ├── models/        # AI models
│   └── training/      # Training logic
├── bin/               # Executable scripts
├── checkpoints/       # Model weights
└── requirements*.txt  # Dependencies
```

### CPU-Only Installation
```bash
# For machines without GPU
pip install -r requirements-cpu.txt
```

## 📋 System Requirements

### 🖥️ Minimum Requirements
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 12+
- **Python**: 3.10 or higher
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB free space
- **Package Manager**: UV (recommended) hoặc pip

### 🚀 Installation Methods Priority
1. **UV + requirements-cpu.txt**: `uv pip install -r requirements-cpu.txt` (CPU-only/Codespaces)
2. **UV + requirements.txt**: `uv pip install -r requirements.txt` (GPU)
3. **pip**: `pip install -r requirements-cpu.txt` hoặc `pip install -r requirements.txt`

## 📚 Documentation

- [📋 README](README.md) - Getting started guide
- [🎥 Demo Guide](docs/DEMO_GUIDE.md) - Usage examples
- [⚙️ Setup Guide](docs/SETUP_GUIDE.md) - Detailed installation
- [🧠 Model Architecture](docs/MODEL_ARCHITECTURE.md) - Technical details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If this project helps you, please give it a star! ⭐**

Made with ❤️ by **DoAnDLL Team**

*🕐 Last updated: July 23, 2025*

</div>
