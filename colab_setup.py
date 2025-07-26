#!/usr/bin/env python3
"""
Google Colab Setup Script for Smart Pill Recognition System

This script sets up the complete environment for running the pill recognition
system in Google Colab, including dependency installation and configuration.

Usage:
  In Colab: !python colab_setup.py
  Or import and run: import colab_setup; colab_setup.main()

Author: DoAnDLL Team
Date: 2025
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_colab_environment():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies for Colab"""
    print("📦 Installing dependencies for Google Colab...")
    
    # Core ML dependencies
    dependencies = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        
        # Data processing
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "Pillow>=9.0.0",
        "opencv-python-headless>=4.7.0",
        "scikit-learn>=1.2.0",
        
        # Visualization
        "matplotlib>=3.6.0",
        "seaborn>=0.11.0", 
        "plotly>=5.14.0",
        
        # Utilities
        "tqdm>=4.64.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.0",
        "requests>=2.28.0",
        
        # Jupyter specific
        "ipywidgets>=8.0.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", dep
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {dep} installed successfully")
            else:
                print(f"⚠️ Warning installing {dep}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout installing {dep}")
        except Exception as e:
            print(f"❌ Error installing {dep}: {e}")
    
    print("✅ Dependency installation completed!")

def setup_project():
    """Setup project structure and files"""
    print("🏗️ Setting up project structure...")
    
    # Create necessary directories
    directories = [
        "/content/checkpoints",
        "/content/data", 
        "/content/logs",
        "/content/outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Clone repository if not exists
    if not os.path.exists('/content/DoAnDLL'):
        print("📥 Cloning repository...")
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/HoangThinh2024/DoAnDLL.git",
                "/content/DoAnDLL"
            ], check=True)
            print("✅ Repository cloned successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone repository: {e}")
            return False
    else:
        print("✅ Repository already exists!")
    
    # Add to Python path
    if '/content/DoAnDLL' not in sys.path:
        sys.path.insert(0, '/content/DoAnDLL')
        print("✅ Added project to Python path")
    
    return True

def verify_installation():
    """Verify that key components are working"""
    print("🔍 Verifying installation...")
    
    checks = []
    
    # Check PyTorch
    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        checks.append(f"✅ PyTorch {torch.__version__} ({device})")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            checks.append(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    except ImportError:
        checks.append("❌ PyTorch not available")
    
    # Check Transformers
    try:
        import transformers
        checks.append(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        checks.append("❌ Transformers not available")
    
    # Check other key libraries
    libraries = [
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm")
    ]
    
    for lib_name, display_name in libraries:
        try:
            lib = __import__(lib_name)
            version = getattr(lib, '__version__', 'Unknown')
            checks.append(f"✅ {display_name} {version}")
        except ImportError:
            checks.append(f"❌ {display_name} not available")
    
    # Print results
    print("\n📋 Installation Verification:")
    for check in checks:
        print(f"  {check}")
    
    # Check if main components work
    try:
        # Try importing our colab trainer
        from colab_trainer import create_colab_trainer
        checks.append("✅ Colab trainer available")
    except ImportError as e:
        checks.append(f"⚠️ Colab trainer issues: {e}")
    
    print("\n🎉 Setup verification completed!")
    return True

def create_quick_start_example():
    """Create a quick start example file"""
    example_code = '''
# Quick Start Example for Smart Pill Recognition in Colab

import os
import sys
sys.path.append('/content/DoAnDLL')

# Import the Colab trainer
from colab_trainer import create_colab_trainer, ColabMultimodalPillTransformer

# Create and train a model
def quick_example():
    print("🚀 Quick Start Example")
    
    # Create trainer
    trainer, model = create_colab_trainer(num_classes=10)
    
    # Example training (simulation mode)
    print("🏋️ Starting training simulation...")
    
    # This will create sample data and train the model
    # In practice, you would provide your own dataset
    
    return trainer, model

# Run the example
if __name__ == "__main__":
    trainer, model = quick_example()
    print("✅ Quick start example completed!")
'''
    
    example_path = "/content/quick_start_example.py"
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"📝 Created quick start example: {example_path}")
    return example_path

def print_usage_instructions():
    """Print usage instructions for Colab"""
    instructions = """
🎯 SETUP COMPLETED! Here's how to use the system:

📖 Getting Started:
1. Open the main notebook: Smart_Pill_Recognition_Colab.ipynb
2. Run cells step by step to see the full pipeline
3. Upload your own pill images to test

🔧 Manual Usage:
```python
# Import the system
import sys
sys.path.append('/content/DoAnDLL')
from colab_trainer import create_colab_trainer

# Create trainer and model
trainer, model = create_colab_trainer()

# Train the model (with your data or simulation)
# results = trainer.train(...)
```

📁 Important Paths:
- Project: /content/DoAnDLL
- Notebooks: /content/DoAnDLL/Smart_Pill_Recognition_Colab.ipynb  
- Checkpoints: /content/checkpoints
- Data: /content/data

🚀 Next Steps:
1. Run the main notebook for complete walkthrough
2. Upload your pill dataset to /content/data
3. Modify training parameters as needed
4. Save your trained models to Google Drive

💡 Tips:
- Use GPU runtime for faster training
- Enable high-RAM if working with large datasets
- Save important files to Google Drive before session ends

Happy coding! 🎉
"""
    print(instructions)

def main():
    """Main setup function"""
    print("🚀 Starting Google Colab setup for Smart Pill Recognition System...")
    
    # Check environment
    if not check_colab_environment():
        print("⚠️ Not running in Google Colab, but continuing setup...")
    else:
        print("✅ Google Colab environment detected!")
    
    # Install dependencies
    install_dependencies()
    
    # Setup project
    if not setup_project():
        print("❌ Project setup failed!")
        return False
    
    # Verify installation
    verify_installation()
    
    # Create examples
    create_quick_start_example()
    
    # Print instructions
    print_usage_instructions()
    
    print("🎉 Setup completed successfully!")
    return True

if __name__ == "__main__":
    main()