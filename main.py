#!/usr/bin/env python3
"""
🚀 Smart Pill Recognition System - Main Launcher
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Launcher script chính cho hệ thống nhận dạng viên thuốc AI
Hỗ trợ cả CLI và Web UI với giao diện đẹp và dễ sử dụng

Tác giả: DoAnDLL Project
Ngày: 2025
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

def show_banner():
    """Hiển thị banner chào mừng"""
    banner = """
\033[96m
   ██████╗ ██╗██╗     ██╗         ██████╗ ███████╗ ██████╗ ███████╗
   ██╔══██╗██║██║     ██║         ██╔══██╗██╔════╝██╔═══██╗██╔════╝
   ██████╔╝██║██║     ██║         ██████╔╝█████╗  ██║   ██║█████╗  
   ██╔═══╝ ██║██║     ██║         ██╔══██╗██╔══╝  ██║   ██║██╔══╝  
   ██║     ██║███████╗███████╗    ██║  ██║███████╗╚██████╔╝███████╗
   ╚═╝     ╚═╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝
\033[0m
\033[93m              🔥 SMART PILL RECOGNITION SYSTEM 🔥\033[0m
\033[2m                AI-Powered Pharmaceutical Identification Platform\033[0m
\033[2m                Tối ưu hóa cho Ubuntu 22.04 + NVIDIA Quadro 6000 + CUDA 12.8\033[0m

"""
    print(banner)

def launch_cli():
    """Khởi chạy CLI"""
    cli_script = PROJECT_ROOT / "apps" / "cli" / "main.py"
    
    if not cli_script.exists():
        print("❌ CLI script không tìm thấy!")
        return False
    
    try:
        subprocess.run([sys.executable, str(cli_script)], cwd=PROJECT_ROOT)
        return True
    except Exception as e:
        print(f"❌ Lỗi khởi chạy CLI: {e}")
        return False

def launch_web():
    """Khởi chạy Web UI"""
    web_script = PROJECT_ROOT / "apps" / "web" / "streamlit_app.py"
    
    if not web_script.exists():
        print("❌ Web UI script không tìm thấy!")
        return False
    
    try:
        # Check if streamlit is installed
        try:
            import streamlit
        except ImportError:
            print("📦 Streamlit chưa được cài đặt. Đang cài đặt...")
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        
        print("🌐 Đang khởi chạy Web UI...")
        print("🔗 Truy cập: http://localhost:8501")
        print("📝 Nhấn Ctrl+C để dừng")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(web_script),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--theme.base", "dark",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#0e1117",
            "--theme.secondaryBackgroundColor", "#262730"
        ]
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng Web UI")
        return True
    except Exception as e:
        print(f"❌ Lỗi khởi chạy Web UI: {e}")
        return False

def recognize_image(image_path, text_imprint=None):
    """Nhận dạng ảnh trực tiếp từ command line"""
    recognize_script = PROJECT_ROOT / "recognize.py"
    
    if not recognize_script.exists():
        print("❌ Recognition script không tìm thấy!")
        return False
    
    try:
        cmd = [sys.executable, str(recognize_script), "--image", image_path]
        if text_imprint:
            cmd.extend(["--text", text_imprint])
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        return True
    except Exception as e:
        print(f"❌ Lỗi nhận dạng: {e}")
        return False

def train_model(config_path=None):
    """Huấn luyện model"""
    train_script = PROJECT_ROOT / "train_cure_model.py"
    
    if not train_script.exists():
        print("❌ Training script không tìm thấy!")
        return False
    
    try:
        cmd = [sys.executable, str(train_script)]
        if config_path:
            cmd.extend(["--config", config_path])
        else:
            cmd.extend(["--config", "config/config.yaml"])
        
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        return True
    except Exception as e:
        print(f"❌ Lỗi training: {e}")
        return False

def setup_environment():
    """Setup environment và dependencies"""
    print("🔧 Đang setup environment...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Yêu cầu Python 3.10+ !")
        return False
    
    # Install requirements
    requirements_file = PROJECT_ROOT / "requirements.txt"
    if requirements_file.exists():
        print("📦 Đang cài đặt dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ Dependencies đã được cài đặt!")
        except subprocess.CalledProcessError:
            print("❌ Lỗi cài đặt dependencies!")
            return False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA: {torch.version.cuda}")
        else:
            print("⚠️  GPU không khả dụng, sẽ sử dụng CPU")
    except ImportError:
        print("📦 PyTorch chưa được cài đặt")
    
    return True

def show_status():
    """Hiển thị trạng thái hệ thống"""
    print("📊 TRẠNG THÁI HỆ THỐNG")
    print("─" * 50)
    
    # Python info
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Project structure
    print(f"📁 Project: {PROJECT_ROOT.name}")
    
    # Check key files
    key_files = [
        "apps/cli/main.py",
        "apps/web/streamlit_app.py", 
        "recognize.py",
        "train_cure_model.py",
        "config/config.yaml",
        "requirements.txt"
    ]
    
    print("\n📄 Key Files:")
    for file_path in key_files:
        full_path = PROJECT_ROOT / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"  {status} {file_path}")
    
    # Check directories
    key_dirs = [
        "Dataset_BigData",
        "checkpoints",
        "core",
        "apps"
    ]
    
    print("\n📂 Directories:")
    for dir_path in key_dirs:
        full_path = PROJECT_ROOT / dir_path
        status = "✅" if full_path.exists() else "❌"
        print(f"  {status} {dir_path}/")
    
    # GPU status
    print("\n🖥️  Hardware:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ Memory: {gpu_memory:.1f} GB")
            print(f"  ✅ CUDA: {torch.version.cuda}")
        else:
            print("  ⚠️  GPU: Không khả dụng")
    except ImportError:
        print("  ❌ PyTorch: Chưa cài đặt")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="🔥 Smart Pill Recognition System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python main.py cli                     # Khởi chạy CLI
  python main.py web                     # Khởi chạy Web UI  
  python main.py recognize image.jpg     # Nhận dạng ảnh
  python main.py train                   # Huấn luyện model
  python main.py setup                   # Setup environment
  python main.py status                  # Kiểm tra trạng thái
        """
    )
    
    parser.add_argument(
        "command",
        choices=["cli", "web", "recognize", "train", "setup", "status"],
        help="Lệnh cần thực thi"
    )
    
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Đường dẫn ảnh (cho lệnh recognize)"
    )
    
    parser.add_argument(
        "--text",
        help="Text imprint trên viên thuốc"
    )
    
    parser.add_argument(
        "--config",
        help="Đường dẫn file config (cho lệnh train)"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Không hiển thị banner"
    )
    
    args = parser.parse_args()
    
    # Show banner unless disabled
    if not args.no_banner:
        show_banner()
    
    # Execute command
    try:
        if args.command == "cli":
            launch_cli()
        
        elif args.command == "web":
            launch_web()
        
        elif args.command == "recognize":
            if not args.image_path:
                print("❌ Cần cung cấp đường dẫn ảnh!")
                print("Ví dụ: python main.py recognize image.jpg")
                return 1
            
            if not Path(args.image_path).exists():
                print(f"❌ File không tồn tại: {args.image_path}")
                return 1
            
            recognize_image(args.image_path, args.text)
        
        elif args.command == "train":
            train_model(args.config)
        
        elif args.command == "setup":
            if not setup_environment():
                return 1
            print("✅ Setup hoàn thành!")
        
        elif args.command == "status":
            show_status()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
