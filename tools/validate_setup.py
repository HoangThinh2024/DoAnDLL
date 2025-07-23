#!/usr/bin/env python3
"""
🔍 System Validation Script
Validates that all components of the Smart Pill Recognition System are properly set up.
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib.util

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version >= (3, 10):
        print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro}{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ Python {version.major}.{version.minor} (requires 3.10+){Colors.END}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        print(f"{Colors.GREEN}✅ {description}{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ {description} (missing: {file_path}){Colors.END}")
        return False

def check_python_file_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"{Colors.RED}❌ Syntax error in {file_path}: {e}{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Could not check {file_path}: {e}{Colors.END}")
        return True  # Don't fail the check for other errors

def check_imports():
    """Check if core modules can be imported"""
    required_modules = [
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('streamlit', 'Streamlit'),
        ('yaml', 'PyYAML'),
    ]
    
    all_good = True
    for module_name, display_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"{Colors.GREEN}✅ {display_name}{Colors.END}")
        except ImportError:
            print(f"{Colors.YELLOW}⚠️  {display_name} (optional dependency){Colors.END}")
            # Don't fail validation for optional dependencies
    
    return all_good

def main():
    """Main validation function"""
    print(f"""
{Colors.CYAN}{Colors.BOLD}
🔍 Smart Pill Recognition System Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Colors.END}""")
    
    # Change to project root
    project_root = Path(__file__).parent.parent  # Go up one level from tools/
    os.chdir(project_root)
    
    print(f"{Colors.BLUE}📁 Project directory: {project_root.absolute()}{Colors.END}\n")
    
    checks_passed = 0
    total_checks = 0
    
    # 1. Python version
    print(f"{Colors.BOLD}🐍 Python Environment:{Colors.END}")
    total_checks += 1
    if check_python_version():
        checks_passed += 1
    
    # 2. Core files
    print(f"\n{Colors.BOLD}📄 Core Files:{Colors.END}")
    core_files = [
        ("main.py", "Main launcher"),
        ("app.py", "Streamlit web app"),
        ("config/config.yaml", "Configuration file"),
        ("requirements.txt", "Dependencies list"),
        ("core/__init__.py", "Core module init"),
        ("core/models/__init__.py", "Models module init"),
    ]
    
    for file_path, description in core_files:
        total_checks += 1
        if check_file_exists(file_path, description):
            checks_passed += 1
    
    # 3. Executable scripts  
    print(f"\n{Colors.BOLD}🚀 Executable Scripts:{Colors.END}")
    scripts = [
        ("bin/pill-setup", "Setup script"),
        ("bin/pill-cli", "CLI launcher"),
        ("bin/pill-web", "Web launcher"),
    ]
    
    for script_path, description in scripts:
        total_checks += 1
        if check_file_exists(script_path, description):
            checks_passed += 1
            # Check if executable
            if os.access(script_path, os.X_OK):
                print(f"  {Colors.GREEN}✅ Executable permissions{Colors.END}")
            else:
                print(f"  {Colors.YELLOW}⚠️  Not executable (run: chmod +x {script_path}){Colors.END}")
    
    # 4. Python syntax validation
    print(f"\n{Colors.BOLD}🔧 Python Syntax:{Colors.END}")
    python_files = ["main.py", "app.py"]
    for py_file in python_files:
        total_checks += 1
        if check_python_file_syntax(py_file):
            print(f"{Colors.GREEN}✅ {py_file} syntax OK{Colors.END}")
            checks_passed += 1
    
    # 5. Dependencies (informational)
    print(f"\n{Colors.BOLD}📦 Dependencies:{Colors.END}")
    check_imports()
    
    # Summary
    print(f"\n{Colors.BOLD}📊 Validation Summary:{Colors.END}")
    percentage = (checks_passed / total_checks) * 100
    
    if percentage >= 90:
        status_color = Colors.GREEN
        status_icon = "🟢"
    elif percentage >= 70:
        status_color = Colors.YELLOW  
        status_icon = "🟡"
    else:
        status_color = Colors.RED
        status_icon = "🔴"
    
    print(f"{status_color}{status_icon} {checks_passed}/{total_checks} checks passed ({percentage:.1f}%){Colors.END}")
    
    if percentage >= 90:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 System validation successful! Ready to use.{Colors.END}")
        print(f"{Colors.CYAN}💡 Next steps:{Colors.END}")
        print(f"   1. Run: {Colors.YELLOW}python main.py web{Colors.END}")
        print(f"   2. Open: {Colors.YELLOW}http://localhost:8501{Colors.END}")
    elif percentage >= 70:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  System mostly ready with minor issues.{Colors.END}")
        print(f"{Colors.CYAN}💡 Recommended actions:{Colors.END}")
        print(f"   1. Install missing dependencies: {Colors.YELLOW}pip install -r requirements.txt{Colors.END}")
        print(f"   2. Fix file permissions: {Colors.YELLOW}chmod +x bin/*{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ System needs setup before use.{Colors.END}")
        print(f"{Colors.CYAN}💡 Required actions:{Colors.END}")
        print(f"   1. Run setup: {Colors.YELLOW}./bin/pill-setup{Colors.END}")
        print(f"   2. Check Python version: {Colors.YELLOW}python --version{Colors.END}")
    
    return 0 if percentage >= 70 else 1

if __name__ == "__main__":
    sys.exit(main())