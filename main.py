#!/usr/bin/env python3
"""
🚀 Smart Pill Recognition System - Main Launcher
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Launcher script chính cho hệ thống nhận dạng viên thuốc AI
Tối ưu hóa cho UV Package Manager với virtual environment .venv

Tác giả: DoAnDLL Project - Pill Recognition Team
Ngày: 2025
Cập nhật: 07/07/2025 21:30 (GMT+7 - Vietnam Time)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import venv
import platform
from datetime import datetime
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Colors for console output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def get_vietnam_time():
    """Lấy thời gian hiện tại theo múi giờ Việt Nam"""
    try:
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        now = datetime.now(vn_tz)
        return now.strftime('%d/%m/%Y %H:%M:%S %Z (GMT%z)')
    except ImportError:
        # Fallback nếu không có pytz
        return datetime.now().strftime('%d/%m/%Y %H:%M:%S (Local Time)')

def show_banner():
    """Hiển thị banner chào mừng với thông tin UV"""
    vn_time = get_vietnam_time()
    banner = f"""
{Colors.CYAN}
   ██████╗ ██╗██╗     ██╗         ██████╗ ███████╗ ██████╗ ███████╗
   ██╔══██╗██║██║     ██║         ██╔══██╗██╔════╝██╔═══██╗██╔════╝
   ██████╔╝██║██║     ██║         ██████╔╝█████╗  ██║   ██║█████╗  
   ██╔═══╝ ██║██║     ██║         ██╔══██╗██╔══╝  ██║   ██║██╔══╝  
   ██║     ██║███████╗███████╗    ██║  ██║███████╗╚██████╔╝███████╗
   ╚═╝     ╚═╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝
{Colors.END}
{Colors.YELLOW}              🔥 SMART PILL RECOGNITION SYSTEM 🔥{Colors.END}
{Colors.WHITE}                AI-Powered Pharmaceutical Identification{Colors.END}
{Colors.MAGENTA}                     🚀 Powered by UV Package Manager 🚀{Colors.END}
{Colors.BLUE}                  📅 Current time: {vn_time}{Colors.END}

"""
    print(banner)

def check_environment():
    """Kiểm tra môi trường và dependencies"""
    print(f"{Colors.BLUE}🔍 Checking environment...{Colors.END}")
    
    issues = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        issues.append(f"❌ Python version {python_version.major}.{python_version.minor} < 3.10 required")
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check UV virtual environment
    venv_path = PROJECT_ROOT / ".venv"
    if not venv_path.exists():
        issues.append("❌ Virtual environment .venv not found")
        print(f"{Colors.YELLOW}💡 Run './bin/pill-setup' or 'make setup' to create environment{Colors.END}")
    else:
        # Check if we're in the virtual environment
        if os.environ.get('VIRTUAL_ENV'):
            print(f"✅ Virtual environment active: {os.environ['VIRTUAL_ENV']}")
        else:
            print(f"{Colors.YELLOW}⚠️  Virtual environment exists but not activated{Colors.END}")
            print(f"{Colors.CYAN}💡 Run 'source .venv/bin/activate' or './activate_env.sh'{Colors.END}")
    
    # Check UV installation
    try:
        uv_result = subprocess.run(['uv', '--version'], capture_output=True, text=True, timeout=5)
        if uv_result.returncode == 0:
            uv_version = uv_result.stdout.strip()
            print(f"✅ UV Package Manager: {uv_version}")
        else:
            issues.append("❌ UV not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        issues.append("❌ UV not found")
        print(f"{Colors.YELLOW}💡 Install UV: curl -LsSf https://astral.sh/uv/install.sh | sh{Colors.END}")
    
    # Check core dependencies
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda} - GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{Colors.YELLOW}⚠️  CUDA not available (CPU mode){Colors.END}")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError:
        issues.append("❌ Streamlit not installed")
    
    if issues:
        print(f"\n{Colors.RED}🚨 Environment Issues Found:{Colors.END}")
        for issue in issues:
            print(f"  {issue}")
        print(f"\n{Colors.CYAN}🔧 Quick Fix:{Colors.END}")
        print(f"  ./bin/pill-setup  # Complete setup")
        print(f"  make setup        # Alternative")
        return False
    
    print(f"{Colors.GREEN}✅ Environment ready!{Colors.END}")
    return True

def launch_cli():
    """Khởi chạy CLI với environment checking"""
    cli_script = PROJECT_ROOT / "apps" / "cli" / "main.py"
    
    if not cli_script.exists():
        print(f"{Colors.RED}❌ CLI script không tìm thấy tại: {cli_script}{Colors.END}")
        return False
    
    print(f"{Colors.BLUE}🖥️ Launching CLI interface...{Colors.END}")
    try:
        # Use current Python executable (should be from venv if activated)
        subprocess.run([sys.executable, str(cli_script)], cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 CLI interface đã được đóng{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Lỗi khởi chạy CLI: {e}{Colors.END}")
        return False
    
    return True

def launch_web():
    """Khởi chạy Web UI với UV environment"""
    web_script = PROJECT_ROOT / "apps" / "web" / "streamlit_app.py"
    
    if not web_script.exists():
        print(f"{Colors.RED}❌ Web UI script không tìm thấy tại: {web_script}{Colors.END}")
        return False
    
    print(f"{Colors.BLUE}🌐 Launching Web UI...{Colors.END}")
    try:
        # Check if streamlit is installed
        try:
            import streamlit
        except ImportError:
            print(f"{Colors.YELLOW}📦 Streamlit chưa được cài đặt. Đang cài đặt với UV...{Colors.END}")
            subprocess.run(['uv', 'pip', 'install', 'streamlit'], check=True)
        
        print(f"{Colors.GREEN}🌐 Starting Streamlit app...{Colors.END}")
        print(f"{Colors.CYAN}🔗 Truy cập: http://localhost:8501{Colors.END}")
        print(f"{Colors.YELLOW}📝 Nhấn Ctrl+C để dừng{Colors.END}")
        
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
        print(f"\n{Colors.YELLOW}� Web UI đã được đóng{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Lỗi khởi chạy Web UI: {e}{Colors.END}")
        return False

def recognize_image(image_path, text_imprint=None):
    """Nhận dạng ảnh trực tiếp từ command line"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"{Colors.RED}❌ Ảnh không tìm thấy: {image_path}{Colors.END}")
        return False
    
    print(f"{Colors.BLUE}🔍 Recognizing pill in: {image_path}{Colors.END}")
    
    try:
        # Import recognition modules
        sys.path.append(str(PROJECT_ROOT / "core"))
        from core.models.multimodal_transformer import PillRecognitionModel
        from core.data.data_processing import preprocess_image
        
        # Load model (if available)
        print(f"{Colors.YELLOW}⏳ Loading model...{Colors.END}")
        
        # For now, just show what would happen
        print(f"{Colors.GREEN}📸 Image: {image_path.name}{Colors.END}")
        if text_imprint:
            print(f"{Colors.GREEN}📝 Text imprint: {text_imprint}{Colors.END}")
        
        print(f"{Colors.CYAN}🔮 Recognition result: [Demo mode - Model loading needed]{Colors.END}")
        print(f"{Colors.YELLOW}💡 Full recognition available after training completion{Colors.END}")
        
        return True
    except ImportError as e:
        print(f"{Colors.RED}❌ Missing dependencies for recognition: {e}{Colors.END}")
        print(f"{Colors.CYAN}💡 Run 'make install' to install all dependencies{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Lỗi nhận dạng: {e}{Colors.END}")
        return False

def setup_environment():
    """Setup environment với UV Package Manager"""
    print(f"{Colors.BLUE}🔧 Setting up environment with UV...{Colors.END}")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"{Colors.RED}❌ Yêu cầu Python 3.10+, tìm thấy {sys.version_info.major}.{sys.version_info.minor}{Colors.END}")
        return False
    
    # Check UV installation
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.YELLOW}📦 UV chưa được cài đặt. Đang cài đặt...{Colors.END}")
        try:
            # Install UV
            subprocess.run([
                'curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh'
            ], shell=True, check=True)
            print(f"{Colors.GREEN}✅ UV đã được cài đặt!{Colors.END}")
        except subprocess.CalledProcessError:
            print(f"{Colors.RED}❌ Không thể cài đặt UV tự động{Colors.END}")
            print(f"{Colors.CYAN}💡 Vui lòng chạy: curl -LsSf https://astral.sh/uv/install.sh | sh{Colors.END}")
            return False
    
    # Run setup script
    setup_script = PROJECT_ROOT / "bin" / "pill-setup"
    if setup_script.exists():
        print(f"{Colors.BLUE}🚀 Running pill-setup script...{Colors.END}")
        try:
            subprocess.run([str(setup_script)], cwd=PROJECT_ROOT, check=True)
            print(f"{Colors.GREEN}✅ Environment setup completed!{Colors.END}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ Setup script failed: {e}{Colors.END}")
            return False
    else:
        print(f"{Colors.YELLOW}⚠️  Setup script not found, using fallback...{Colors.END}")
        # Fallback setup
        try:
            # Create virtual environment
            venv_path = PROJECT_ROOT / ".venv"
            if not venv_path.exists():
                print(f"{Colors.BLUE}📦 Creating virtual environment...{Colors.END}")
                subprocess.run(['uv', 'venv', '.venv', '--python', '3.10'], 
                             cwd=PROJECT_ROOT, check=True)
            
            # Install dependencies
            print(f"{Colors.BLUE}📚 Installing dependencies...{Colors.END}")
            subprocess.run(['uv', 'pip', 'install', '-e', '.'], 
                         cwd=PROJECT_ROOT, check=True)
            
            print(f"{Colors.GREEN}✅ Fallback setup completed!{Colors.END}")
            print(f"{Colors.CYAN}💡 Activate environment: source .venv/bin/activate{Colors.END}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}❌ Fallback setup failed: {e}{Colors.END}")
            return False

def train_model(config_path=None):
    """Khởi chạy quá trình huấn luyện model"""
    trainer_script = PROJECT_ROOT / "core" / "training" / "trainer.py"
    
    if not trainer_script.exists():
        print(f"{Colors.RED}❌ Training script không tìm thấy tại: {trainer_script}{Colors.END}")
        return False
    
    print(f"{Colors.BLUE}🏋️ Starting model training...{Colors.END}")
    print(f"{Colors.YELLOW}⚠️  This may take several hours depending on your hardware{Colors.END}")
    
    try:
        cmd = [sys.executable, str(trainer_script)]
        if config_path:
            cmd.extend(['--config', config_path])
        subprocess.run(cmd, cwd=PROJECT_ROOT)
        return True
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⏹️ Training interrupted by user{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Training error: {e}{Colors.END}")
        return False

def show_status():
    """Hiển thị trạng thái hệ thống"""
    vn_time = get_vietnam_time()
    print(f"{Colors.BLUE}📋 System Status{Colors.END}")
    print("=" * 50)
    print(f"🕐 Current time: {vn_time}")
    
    # Python info
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Project: {PROJECT_ROOT}")
    
    # Virtual environment
    venv_path = PROJECT_ROOT / ".venv"
    if venv_path.exists():
        print(f"🗂️  Virtual Env: ✅ {venv_path}")
        if os.environ.get('VIRTUAL_ENV'):
            print(f"🔄 Status: ✅ Activated")
        else:
            print(f"🔄 Status: ⚠️  Not activated")
    else:
        print(f"🗂️  Virtual Env: ❌ Not found")
    
    # UV
    try:
        uv_result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if uv_result.returncode == 0:
            print(f"📦 UV: ✅ {uv_result.stdout.strip()}")
        else:
            print(f"📦 UV: ❌ Not working")
    except FileNotFoundError:
        print(f"📦 UV: ❌ Not found")
    
    # GPU
    try:
        gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
        if gpu_result.returncode == 0:
            gpu_name = gpu_result.stdout.strip().split('\n')[0]
            print(f"🎮 GPU: ✅ {gpu_name}")
        else:
            print(f"🎮 GPU: ❌ Not detected")
    except FileNotFoundError:
        print(f"🎮 GPU: ❌ nvidia-smi not found")
    
    # Dependencies
    print(f"\n{Colors.BLUE}📦 Key Dependencies:{Colors.END}")
    deps_to_check = ['torch', 'torchvision', 'streamlit', 'pandas', 'numpy', 'PIL']
    
    for dep in deps_to_check:
        try:
            if dep == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(dep)
                version = getattr(module, '__version__', 'Unknown')
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ❌ {dep}: Not installed")
    
    print(f"\n{Colors.GREEN}💡 Use 'python main.py setup' if issues found{Colors.END}")
    print(f"📅 Status checked at: {vn_time}")

def main():
    """Main function với full UV support"""
    parser = argparse.ArgumentParser(
        description='🚀 Smart Pill Recognition System - UV Powered',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Colors.CYAN}Examples:{Colors.END}
  {Colors.YELLOW}python main.py cli{Colors.END}                     # Launch CLI interface
  {Colors.YELLOW}python main.py web{Colors.END}                     # Launch web interface  
  {Colors.YELLOW}python main.py recognize image.jpg{Colors.END}      # Recognize pill
  {Colors.YELLOW}python main.py recognize img.jpg --text "P500"{Colors.END}  # With text
  {Colors.YELLOW}python main.py setup{Colors.END}                   # Setup environment
  {Colors.YELLOW}python main.py train{Colors.END}                   # Train model
  {Colors.YELLOW}python main.py status{Colors.END}                  # Show system status

{Colors.CYAN}Environment:{Colors.END}
  📦 Package Manager: UV (https://github.com/astral-sh/uv)
  🐍 Python: 3.10+ required
  🗂️  Virtual Env: .venv (auto-created)
  🚀 Setup: ./bin/pill-setup or make setup

{Colors.GREEN}🌟 Happy coding with Pill Recognition System! 🌟{Colors.END}
        """
    )
    
    parser.add_argument(
        'command',
        choices=['cli', 'web', 'recognize', 'train', 'setup', 'status', 'check'],
        help='Lệnh để thực thi'
    )
    
    parser.add_argument(
        'image_path',
        nargs='?',
        help='Đường dẫn ảnh (cho lệnh recognize)'
    )
    
    parser.add_argument(
        '--text', '-t',
        help='Text imprint trên viên thuốc'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Đường dẫn config file (cho training)'
    )
    
    parser.add_argument(
        '--no-banner',
        action='store_true',
        help='Không hiển thị banner'
    )
    
    parser.add_argument(
        '--env-check',
        action='store_true',
        help='Kiểm tra environment trước khi chạy'
    )
    
    args = parser.parse_args()
    
    # Show banner
    if not args.no_banner:
        show_banner()
    
    # Environment check
    if args.env_check or args.command in ['cli', 'web', 'train']:
        if not check_environment():
            print(f"\n{Colors.YELLOW}💡 Environment issues detected. Run 'python main.py setup' first{Colors.END}")
            return 1
    
    # Execute commands
    success = True
    
    if args.command == 'cli':
        success = launch_cli()
    
    elif args.command == 'web':
        success = launch_web()
    
    elif args.command == 'recognize':
        if not args.image_path:
            print(f"{Colors.RED}❌ Vui lòng cung cấp đường dẫn ảnh{Colors.END}")
            print(f"{Colors.CYAN}💡 Sử dụng: python main.py recognize path/to/image.jpg{Colors.END}")
            return 1
        success = recognize_image(args.image_path, args.text)
    
    elif args.command == 'train':
        success = train_model(args.config)
    
    elif args.command == 'setup':
        success = setup_environment()
        if success:
            print(f"\n{Colors.GREEN}🎉 Setup completed successfully!{Colors.END}")
            print(f"{Colors.CYAN}Next steps:{Colors.END}")
            print(f"  1. Activate environment: {Colors.YELLOW}source .venv/bin/activate{Colors.END}")
            print(f"  2. Run web app: {Colors.YELLOW}python main.py web{Colors.END}")
            print(f"  3. Or CLI: {Colors.YELLOW}python main.py cli{Colors.END}")
    
    elif args.command == 'status':
        show_status()
    
    elif args.command == 'check':
        success = check_environment()
    
    if not success:
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        sys.exit(1)
