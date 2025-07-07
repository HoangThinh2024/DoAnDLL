# 🚀 Quick Setup Guide - Hướng dẫn cài đặt nhanh

*📅 Last updated: 07/07/2025 21:30 (GMT+7 - Vietnam Time)*

## ⚡ Cài đặt siêu nhanh (1 lệnh)

```bash
# Clone và setup toàn bộ
git clone <repository-url>
cd DoAnDLL
./bin/pill-setup
```

## 📋 Chi tiết từng bước

### 1. 🔧 Chuẩn bị hệ thống

```bash
# Ubuntu/Debian - Cài đặt Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev curl

# Kiểm tra Python
python3 --version  # Phải >= 3.10
```

### 2. 📦 Cài đặt UV Package Manager

```bash
# Cài đặt UV (tự động)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Kiểm tra
uv --version
```

### 3. 🏗️ Setup dự án

```bash
# Tự động (Khuyến nghị)
./bin/pill-setup

# Hoặc thủ công
make setup
```

### 4. 🚀 Kích hoạt và sử dụng

```bash
# Kích hoạt môi trường
source .venv/bin/activate
# hoặc
./activate_env.sh

# Chạy ứng dụng
./bin/pill-web    # Web interface
./bin/pill-cli    # CLI interface
```

---

## 🎯 Sử dụng nhanh

### 🌐 Web Interface (Khuyến nghị cho người mới)

```bash
./bin/pill-web
# ➡️ Mở http://localhost:8501
```

### 🖥️ CLI Interface (Cho developer)

```bash
./bin/pill-cli
```

### 🔍 Nhận dạng trực tiếp

```bash
# Nhận dạng ảnh
python main.py recognize path/to/image.jpg

# Với text trên viên thuốc
python main.py recognize image.jpg --text "P500"

# Xem tất cả options
python main.py --help
```

---

## 🛠️ Commands hữu ích

### 📦 Package Management với UV

```bash
# Cài đặt package mới
uv pip install package-name

# Cập nhật packages
uv pip install --upgrade package-name

# Liệt kê packages
uv pip list

# Kiểm tra conflicts
uv pip check
```

### 🧪 Testing & Development

```bash
# Chạy tests
make test

# Format code
make format

# Lint code  
make lint

# Jupyter notebook
make notebook
```

### 📊 Monitoring

```bash
# Trạng thái hệ thống
make status

# Monitor GPU
make monitor

# Thông tin project
make info
```

---

## 🔧 Troubleshooting

### ❌ Lỗi thường gặp

#### "UV not found"
```bash
# Cài đặt lại UV
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

#### "Python version too old"
```bash
# Ubuntu - Cài Python 3.10+
sudo apt install python3.10 python3.10-venv
```

#### "Virtual environment not found"
```bash
# Tạo lại môi trường
make clean-env
make setup-env
```

#### "CUDA not available"
```bash
# Kiểm tra GPU
nvidia-smi

# Cài đặt CUDA dependencies
make install-gpu
```

### 🔍 Kiểm tra cài đặt

```bash
# Kiểm tra tất cả
make status

# Test dependencies
python -c "
import torch
import streamlit  
import pandas
print('✅ All core packages available')
"
```

---

## 📚 Next Steps

1. 📖 Đọc [README.md](README.md) đầy đủ
2. 🎥 Xem [Demo Guide](docs/DEMO_GUIDE.md)  
3. 🏗️ Khám phá [Project Structure](docs/PROJECT_STRUCTURE.md)
4. 🧠 Tìm hiểu [Model Architecture](docs/MODEL_ARCHITECTURE.md)

---

## 💡 Tips

- 🔥 **Luôn kích hoạt venv** trước khi làm việc: `source .venv/bin/activate`
- 📦 **Sử dụng UV** cho package management (nhanh hơn pip)
- 🧹 **Dọn dẹp thường xuyên**: `make clean`  
- 📊 **Monitor GPU** khi training: `make monitor`
- 🚀 **Web interface** dễ dùng nhất cho người mới

---

---

## 📅 Update History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 07/07/2025 21:30 (GMT+7) | UV integration, enhanced setup |
| 1.0.0 | Previous | Initial pip-based setup |

<div align="center">

**🌟 Happy coding with Pill Recognition System! 🌟**

*📅 Guide updated: 07/07/2025 21:30 (Vietnam Time)*

</div>
