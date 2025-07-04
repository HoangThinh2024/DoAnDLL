#!/bin/bash

# Script khởi chạy hệ thống nhận dạng viên thuốc

echo "🚀 Khởi động Hệ thống Nhận dạng Viên Thuốc Multimodal"
echo "================================================="

# Kiểm tra Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $python_version"

# Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits
else
    echo "⚠️  No GPU detected, using CPU"
fi

# Kiểm tra dependencies
echo "🔍 Kiểm tra dependencies..."
pip check > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Dependencies OK"
else
    echo "❌ Dependencies có vấn đề, đang cài đặt..."
    pip install -r requirements.txt
fi

# Tạo thư mục cần thiết
echo "📁 Tạo thư mục cần thiết..."
mkdir -p data/raw data/processed checkpoints logs results

# Khởi động Streamlit
echo "🌐 Khởi động giao diện web..."
echo "Truy cập: http://localhost:8501"
echo "Nhấn Ctrl+C để dừng"

streamlit run app.py
