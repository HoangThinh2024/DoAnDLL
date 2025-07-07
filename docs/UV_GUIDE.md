# 📦 UV Package Manager Guide
*Hướng dẫn sử dụng UV cho Pill Recognition System*

---
**📅 Cập nhật lần cuối:** 07/07/2025 - 14:30 GMT+7 (Múi giờ Việt Nam)  
**📋 Phiên bản:** v2.0  
**⏰ Thời gian hiện tại:** 07/07/2025 - 14:30 ICT (UTC+7)
---

## 🚀 Tại sao sử dụng UV?

**UV** là package manager Python hiện đại, nhanh hơn pip 10-100x:

- ⚡ **Siêu nhanh**: Cài đặt dependencies trong giây lát
- 🔒 **Dependency resolution**: Giải quyết conflicts tự động  
- 📦 **Compatible**: Hoạt động với pip, pipenv, poetry
- 🐍 **Multiple Python**: Quản lý nhiều phiên bản Python
- 🌐 **Cross-platform**: Linux, Windows, macOS

## 📋 Installation

### Linux/macOS (Khuyến nghị)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Manual Install
```bash
pip install uv
```

## 🏗️ Project Setup

### 1. Tạo Virtual Environment
```bash
# Tạo venv với Python 3.10
uv venv .venv --python 3.10

# Kích hoạt
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 2. Cài đặt Dependencies
```bash
# Từ pyproject.toml (khuyến nghị)
uv pip install -e .

# Từ requirements.txt
uv pip install -r requirements.txt

# Package cụ thể
uv pip install torch torchvision
```

### 3. Optional Dependencies
```bash
# GPU support
uv pip install -e ".[gpu]"

# Development tools
uv pip install -e ".[dev]"

# Tất cả
uv pip install -e ".[all]"
```

## 🔧 Quản lý Dependencies

### Cài đặt Packages
```bash
# Cài package mới
uv pip install package-name

# Với version cụ thể
uv pip install "torch>=2.0,<3.0"

# Từ git
uv pip install git+https://github.com/user/repo.git

# Local editable
uv pip install -e .
```

### Cập nhật Packages
```bash
# Cập nhật 1 package
uv pip install --upgrade torch

# Cập nhật tất cả
uv pip install --upgrade -r requirements.txt

# Cập nhật project
uv pip install -e . --upgrade
```

### Liệt kê & Kiểm tra
```bash
# Danh sách packages
uv pip list

# Chi tiết package
uv pip show torch

# Kiểm tra conflicts
uv pip check

# Export requirements
uv pip freeze > requirements.txt
```

## 🚀 Commands cho Pill Recognition

### Setup Project
```bash
# Cài đặt hoàn chỉnh
./bin/pill-setup

# Hoặc thủ công
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Development
```bash
# Dev environment
uv pip install -e ".[dev]"

# Thêm package
uv pip install jupyter
uv pip install pytest

# Format & Lint
uv pip install black isort flake8
```

### GPU Support
```bash
# CUDA 12.1 (khuyến nghị)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA packages
uv pip install cudf-cu12 cuml-cu12 cupy-cuda12x
```

## 📊 So sánh Performance

| Tool | Time | Memory | Disk |
|------|------|--------|------|
| **uv** | 0.5s | 10MB | Fast |
| pip | 30s | 50MB | Slow |
| conda | 120s | 200MB | Very Slow |

*Cài đặt PyTorch + dependencies (~500MB)*

## 🔍 Troubleshooting

### UV không tìm thấy
```bash
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc

# Restart shell
source ~/.bashrc
```

### Conflicts Resolution
```bash
# Kiểm tra conflicts
uv pip check

# Force reinstall
uv pip install --force-reinstall torch

# Clean install
rm -rf .venv
uv venv .venv --python 3.10
uv pip install -e .
```

### Python Version Issues
```bash
# List available Python
uv python list

# Install specific Python
uv python install 3.10

# Use specific Python
uv venv .venv --python 3.10.12
```

## 🎯 Best Practices

### 1. **Luôn sử dụng Virtual Environment**
```bash
# Tạo venv cho mỗi project
uv venv .venv --python 3.10
source .venv/bin/activate
```

### 2. **Pin Dependencies**
```toml
# pyproject.toml
dependencies = [
    "torch>=2.3.0,<3.0",
    "streamlit>=1.32.0,<2.0"
]
```

### 3. **Sử dụng Optional Dependencies**
```toml
[project.optional-dependencies]
gpu = ["cudf-cu12>=24.04.0"]
dev = ["pytest>=7.4.0"]
```

### 4. **Regular Updates**
```bash
# Weekly update check
uv pip list --outdated

# Update critical packages
uv pip install --upgrade torch streamlit
```

## 🔧 Advanced Usage

### Multiple Python Versions
```bash
# Install Python 3.11
uv python install 3.11

# Create venv with 3.11
uv venv .venv-311 --python 3.11

# Switch between versions
source .venv/bin/activate      # Python 3.10
source .venv-311/bin/activate  # Python 3.11
```

### Custom Index
```bash
# PyTorch CUDA index
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# Private index
uv pip install my-package --index-url https://my-private-index.com/simple
```

### Dependency Constraints
```bash
# With constraints file
uv pip install -r requirements.txt -c constraints.txt

# No dependencies
uv pip install torch --no-deps
```

## 📚 Resources

- 📖 [UV Documentation](https://docs.astral.sh/uv/)
- 🚀 [GitHub Repository](https://github.com/astral-sh/uv)
- 💬 [Community Discord](https://discord.gg/astral-sh)
- 🐛 [Issue Tracker](https://github.com/astral-sh/uv/issues)

## 🤝 Integration với Pill Recognition

### Make Commands
```bash
make setup       # UV setup + venv + deps
make install     # Core dependencies
make install-gpu # GPU packages
make clean-env   # Remove venv
```

### Python Scripts
```bash
python main.py setup    # Auto UV setup
python main.py status   # Check UV status
python main.py check    # Environment check
```

### Docker Integration
```dockerfile
# UV trong Docker
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"
RUN uv pip install -r requirements.txt
```

---

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 07/07/2025 21:30 (GMT+7) | 🚀 Initial UV integration, complete rewrite |
| 1.0.0 | Previous | Legacy pip-based system |

---

<div align="center">

**⚡ UV makes Python dependency management blazingly fast! ⚡**

*Made with ❤️ for Pill Recognition System*

*📅 Last updated: 07/07/2025 21:30 (GMT+7 - Vietnam Time)*

</div>
