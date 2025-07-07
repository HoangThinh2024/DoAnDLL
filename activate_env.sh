#!/bin/bash

# 🚀 Activate Pill Recognition Environment
# Kích hoạt môi trường ảo cho hệ thống nhận dạng viên thuốc

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Banner
echo -e "${BLUE}"
cat << "EOF"
┌─────────────────────────────────────────┐
│  💊 Smart Pill Recognition System      │
│  🚀 Activating Environment...          │
└─────────────────────────────────────────┘
EOF
echo -e "${NC}"

# Show current Vietnam time
echo -e "${BLUE}🕐 Activation time: $(TZ='Asia/Ho_Chi_Minh' date '+%d/%m/%Y %H:%M:%S %Z (GMT%z)')${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}💡 Run './bin/pill-setup' or 'make setup' first${NC}"
    exit 1
fi

# Activate environment
echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source .venv/bin/activate

# Check activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Environment activated successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 Environment Info:${NC}"
    echo "────────────────────────"
    echo "🐍 Python: $(python --version)"
    echo "📁 Virtual Env: $VIRTUAL_ENV"
    echo "📦 Packages: $(pip list | wc -l) installed"
    
    # Check key packages
    echo ""
    echo -e "${BLUE}🔍 Key Dependencies:${NC}"
    echo "─────────────────────────"
    
    python -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('🎮 CUDA Available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('🔥 GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('❌ PyTorch: Not installed')

try:
    import streamlit as st
    print('✅ Streamlit:', st.__version__)
except ImportError:
    print('❌ Streamlit: Not installed')

try:
    import pandas as pd
    import numpy as np
    from PIL import Image
    print('✅ Data Science packages: Available')
except ImportError:
    print('❌ Some data science packages missing')
"
    
    echo ""
    echo -e "${GREEN}🚀 Ready to use! Available commands:${NC}"
    echo "────────────────────────────────────────"
    echo "🖥️  ./bin/pill-cli          # CLI interface"
    echo "🌐 ./bin/pill-web          # Web interface"
    echo "🔍 python main.py recognize image.jpg"
    echo "🏋️  python main.py train"
    echo "📊 make status             # System status"
    echo "❓ make help               # All commands"
    echo ""
    echo -e "${YELLOW}💡 To deactivate: type 'deactivate'${NC}"
    echo -e "${BLUE}📅 Environment activated at: $(TZ='Asia/Ho_Chi_Minh' date '+%d/%m/%Y %H:%M:%S %Z (GMT%z)')${NC}"
    
else
    echo -e "${RED}❌ Failed to activate environment${NC}"
    exit 1
fi
