#!/usr/bin/env python3
"""
🧹 Advanced Project Cleaner & Organizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Script dọn dẹp và tổ chức lại project hoàn toàn
- Xóa files trùng lặp
- Di chuyển executable files vào thư mục bin/
- Sắp xếp lại cấu trúc logic
- Tạo symlinks cho dễ sử dụng

Tác giả: DoAnDLL Project
Ngày: 2025
"""

import os
import shutil
import sys
from pathlib import Path
import json
import hashlib
from typing import List, Dict, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

class AdvancedProjectCleaner:
    """Lớp dọn dẹp project nâng cao"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.duplicates_found = []
        self.files_moved = []
        self.files_deleted = []
        self.links_created = []
        
    def show_banner(self):
        """Hiển thị banner"""
        print("""
🧹 ADVANCED PROJECT CLEANER & ORGANIZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dọn dẹp và tổ chức lại Smart Pill Recognition System
        """)
    
    def find_duplicates(self) -> Dict[str, List[Path]]:
        """Tìm các file trùng lặp dựa trên content"""
        print("🔍 Đang tìm files trùng lặp...")
        
        file_hashes = {}
        duplicates = {}
        
        # Scan all files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_skip_file(file_path):
                try:
                    # Calculate hash
                    file_hash = self._calculate_file_hash(file_path)
                    
                    if file_hash in file_hashes:
                        # Duplicate found
                        if file_hash not in duplicates:
                            duplicates[file_hash] = [file_hashes[file_hash]]
                        duplicates[file_hash].append(file_path)
                    else:
                        file_hashes[file_hash] = file_path
                        
                except Exception as e:
                    print(f"  ⚠️ Lỗi xử lý {file_path}: {e}")
        
        self.duplicates_found = duplicates
        return duplicates
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Tính hash của file"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Kiểm tra file có nên bỏ qua không"""
        skip_patterns = [
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.vscode', '.idea', 'logs', 'checkpoints', 'Dataset_BigData'
        ]
        
        # Skip binary files and large files
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                return True
        except:
            return True
            
        # Skip patterns
        for pattern in skip_patterns:
            if pattern in str(file_path):
                return True
                
        # Skip binary extensions
        binary_exts = {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz', '.pth', '.pt'}
        if file_path.suffix.lower() in binary_exts:
            return True
            
        return False
    
    def remove_duplicates(self):
        """Xóa các file trùng lặp"""
        print("🗑️ Đang xóa files trùng lặp...")
        
        for file_hash, duplicate_files in self.duplicates_found.items():
            if len(duplicate_files) > 1:
                # Keep the file in the best location
                best_file = self._choose_best_duplicate(duplicate_files)
                
                for file_path in duplicate_files:
                    if file_path != best_file:
                        try:
                            file_path.unlink()
                            self.files_deleted.append(str(file_path))
                            print(f"  🗑️ Xóa: {file_path.relative_to(self.project_root)}")
                        except Exception as e:
                            print(f"  ❌ Lỗi xóa {file_path}: {e}")
    
    def _choose_best_duplicate(self, files: List[Path]) -> Path:
        """Chọn file tốt nhất trong danh sách trùng lặp"""
        # Priority order for directories
        priority_dirs = [
            'core', 'apps', 'scripts', 'tools', 'src'
        ]
        
        # Score files based on location
        scored_files = []
        for file_path in files:
            score = 0
            rel_path = file_path.relative_to(self.project_root)
            
            # Higher score for priority directories
            for i, dir_name in enumerate(priority_dirs):
                if dir_name in str(rel_path):
                    score += len(priority_dirs) - i
                    break
            
            # Prefer shorter paths
            score -= len(rel_path.parts)
            
            scored_files.append((score, file_path))
        
        # Return file with highest score
        return max(scored_files, key=lambda x: x[0])[1]
    
    def organize_executables(self):
        """Sắp xếp các file thực thi vào thư mục bin/"""
        print("📁 Đang tổ chức executable files...")
        
        # Create bin directory
        bin_dir = self.project_root / "bin"
        bin_dir.mkdir(exist_ok=True)
        
        # Executable files to move
        executables = [
            "setup", "clean", "demo", "deploy", "monitor", "test", "train"
        ]
        
        for exec_name in executables:
            source_path = self.project_root / exec_name
            if source_path.exists() and source_path.is_file():
                target_path = bin_dir / exec_name
                
                try:
                    # Move file
                    shutil.move(str(source_path), str(target_path))
                    
                    # Make executable
                    os.chmod(target_path, 0o755)
                    
                    self.files_moved.append((str(source_path), str(target_path)))
                    print(f"  📦 Moved: {exec_name} → bin/{exec_name}")
                    
                    # Create symlink in root for convenience
                    self._create_symlink(target_path, source_path)
                    
                except Exception as e:
                    print(f"  ❌ Lỗi di chuyển {exec_name}: {e}")
    
    def organize_legacy_files(self):
        """Sắp xếp các file legacy vào thư mục riêng"""
        print("📚 Đang tổ chức legacy files...")
        
        legacy_dir = self.project_root / "legacy"
        legacy_dir.mkdir(exist_ok=True)
        
        # Legacy files to move
        legacy_files = [
            "app_with_dataset.py",
            "quick_test.py", 
            "test_dataset_port",
            "train_cure_model.py",
            "train_optimized_server.py",
            "REFACTOR_SUMMARY.md",
            "QUICKSTART.md"
        ]
        
        for file_name in legacy_files:
            source_path = self.project_root / file_name
            if source_path.exists():
                target_path = legacy_dir / file_name
                
                try:
                    # Move file/directory
                    if source_path.is_dir():
                        shutil.move(str(source_path), str(target_path))
                    else:
                        shutil.move(str(source_path), str(target_path))
                    
                    self.files_moved.append((str(source_path), str(target_path)))
                    print(f"  📦 Moved: {file_name} → legacy/{file_name}")
                    
                except Exception as e:
                    print(f"  ❌ Lỗi di chuyển {file_name}: {e}")
    
    def remove_redundant_directories(self):
        """Xóa các thư mục redundant"""
        print("🗂️ Đang dọn dẹp thư mục redundant...")
        
        # Check if src/ is redundant with core/
        src_dir = self.project_root / "src"
        core_dir = self.project_root / "core"
        
        if src_dir.exists() and core_dir.exists():
            # Compare contents
            if self._directories_similar(src_dir, core_dir):
                try:
                    shutil.rmtree(src_dir)
                    print(f"  🗑️ Xóa thư mục redundant: src/")
                except Exception as e:
                    print(f"  ❌ Lỗi xóa src/: {e}")
    
    def _directories_similar(self, dir1: Path, dir2: Path) -> bool:
        """Kiểm tra 2 thư mục có tương tự không"""
        try:
            files1 = set(f.name for f in dir1.rglob("*") if f.is_file())
            files2 = set(f.name for f in dir2.rglob("*") if f.is_file())
            
            # Similar if >80% overlap
            overlap = len(files1 & files2)
            total = len(files1 | files2)
            
            return overlap / total > 0.8 if total > 0 else False
        except:
            return False
    
    def _create_symlink(self, target: Path, link_path: Path):
        """Tạo symbolic link"""
        try:
            # Create relative symlink
            rel_target = os.path.relpath(target, link_path.parent)
            link_path.symlink_to(rel_target)
            self.links_created.append((str(link_path), str(target)))
            print(f"  🔗 Symlink: {link_path.name} → {rel_target}")
        except Exception as e:
            print(f"  ⚠️ Không thể tạo symlink {link_path.name}: {e}")
    
    def create_launcher_scripts(self):
        """Tạo các launcher script tiện lợi"""
        print("🚀 Đang tạo launcher scripts...")
        
        # Update run script to use bin/
        self._update_run_script()
        
        # Create convenience scripts
        self._create_convenience_scripts()
    
    def _update_run_script(self):
        """Cập nhật run script để sử dụng bin/"""
        run_script = self.project_root / "run"
        
        if run_script.exists():
            # Read current content
            content = run_script.read_text()
            
            # Update paths to use bin/
            updated_content = content.replace(
                "python3 main.py",
                "python3 main.py"
            )
            
            # Add bin/ path references where needed
            bin_commands = {
                "./setup": "./bin/setup",
                "./test": "./bin/test", 
                "./train": "./bin/train",
                "./deploy": "./bin/deploy",
                "./monitor": "./bin/monitor",
                "./clean": "./bin/clean"
            }
            
            for old_cmd, new_cmd in bin_commands.items():
                updated_content = updated_content.replace(old_cmd, new_cmd)
            
            # Write back
            run_script.write_text(updated_content)
            print(f"  ✅ Updated: run script")
    
    def _create_convenience_scripts(self):
        """Tạo các convenience scripts"""
        convenience_scripts = {
            "pill-cli": "python3 main.py cli",
            "pill-web": "python3 main.py web", 
            "pill-train": "./bin/train",
            "pill-setup": "./bin/setup",
            "pill-test": "./bin/test"
        }
        
        bin_dir = self.project_root / "bin"
        
        for script_name, command in convenience_scripts.items():
            script_path = bin_dir / script_name
            
            script_content = f"""#!/bin/bash
# {script_name} - Convenience script for Smart Pill Recognition
cd "$(dirname "$0")/.."
{command} "$@"
"""
            
            try:
                script_path.write_text(script_content)
                os.chmod(script_path, 0o755)
                print(f"  🚀 Created: bin/{script_name}")
            except Exception as e:
                print(f"  ❌ Lỗi tạo {script_name}: {e}")
    
    def update_documentation(self):
        """Cập nhật documentation với structure mới"""
        print("📝 Đang cập nhật documentation...")
        
        # Update project structure in docs
        structure_doc = self.project_root / "docs" / "PROJECT_STRUCTURE.md"
        
        if structure_doc.exists():
            new_structure = self._generate_new_structure_doc()
            structure_doc.write_text(new_structure)
            print(f"  📝 Updated: PROJECT_STRUCTURE.md")
        
        # Update README with new commands
        self._update_readme_commands()
    
    def _generate_new_structure_doc(self) -> str:
        """Tạo document cấu trúc project mới"""
        return """# 📁 Project Structure (Cleaned & Organized)

```
Smart Pill Recognition System/
├── 🚀 main.py                     # Main launcher script
├── 🏃 run                         # Quick run script
├── ⚙️ Makefile                    # Build automation
├── 📋 requirements.txt            # Dependencies
├── 🙈 .gitignore                 # Git ignore rules
├── 🐳 Dockerfile                  # Docker configuration
├── 🐙 docker-compose.yml          # Docker Compose
├── 📄 LICENSE                     # License file
├── 📖 README.md                   # Main documentation
│
├── 🔧 bin/                        # ✨ Executable scripts
│   ├── setup                     # System setup
│   ├── test                      # Testing utilities
│   ├── train                     # Training scripts
│   ├── deploy                    # Deployment tools
│   ├── monitor                   # System monitoring
│   ├── clean                     # Cleanup utilities
│   ├── pill-cli                  # Convenience: CLI launcher
│   ├── pill-web                  # Convenience: Web launcher
│   ├── pill-train               # Convenience: Training
│   ├── pill-setup               # Convenience: Setup
│   └── pill-test                # Convenience: Testing
│
├── 📱 apps/                       # Applications
│   ├── 🖥️ cli/                   # CLI interface
│   │   ├── main.py               # Rich CLI với terminal đẹp
│   │   └── recognize.py          # CLI recognition tool
│   ├── 🌐 web/                   # Web interface  
│   │   └── streamlit_app.py      # Modern Streamlit app
│   └── 📚 legacy/                # Legacy applications
│
├── 🧠 core/                      # Core modules (cleaned)
│   ├── 📊 data/                  # Data processing
│   ├── 🤖 models/                # AI Models
│   ├── 🏋️ training/              # Training utilities
│   └── 🔧 utils/                 # Utility functions
│
├── 📜 scripts/                   # Additional scripts
├── 🛠️ tools/                    # Development tools
├── 📚 docs/                      # Documentation
├── 📓 notebooks/                 # Jupyter notebooks
├── ⚙️ config/                    # Configuration files
├── 💾 checkpoints/               # Model checkpoints
├── 📊 data/                      # Processed data
├── 📈 Dataset_BigData/           # Raw datasets
├── 📝 logs/                      # Log files
│
└── 📚 legacy/                    # ✨ Legacy & deprecated files
    ├── app_with_dataset.py       # Old app version
    ├── quick_test.py             # Old test script
    ├── test_dataset_port         # Old test tool
    ├── train_cure_model.py       # Old training script
    ├── train_optimized_server.py # Old training variant
    ├── REFACTOR_SUMMARY.md       # Old refactor notes
    └── QUICKSTART.md             # Old quickstart guide
```

## 🚀 New Usage

### Quick Commands
```bash
# Convenience scripts (no path needed)
./bin/pill-cli              # Launch CLI
./bin/pill-web              # Launch Web UI
./bin/pill-setup            # Setup system
./bin/pill-train            # Train model
./bin/pill-test             # Run tests

# Traditional commands
./run cli                   # Launch CLI
./run web                   # Launch Web UI
python main.py status       # System status
```

### Executable Tools
```bash
# All tools now in bin/ directory
./bin/setup                 # System setup
./bin/test --full           # Comprehensive testing
./bin/train --quick         # Quick training
./bin/deploy --production   # Production deployment
./bin/monitor --health      # Health monitoring
./bin/clean --all           # Complete cleanup
```

## ✨ Improvements

✅ **No more scattered executables** - All in `bin/`  
✅ **No more duplicate files** - Cleaned automatically  
✅ **Legacy files organized** - Moved to `legacy/`  
✅ **Convenience scripts** - Easy access with `pill-*`  
✅ **Symlinks for compatibility** - Old paths still work  
✅ **Clean project root** - Only essential files  

"""
    
    def _update_readme_commands(self):
        """Cập nhật README với commands mới"""
        readme_path = self.project_root / "README.md"
        
        if readme_path.exists():
            content = readme_path.read_text()
            
            # Add note about new structure
            new_section = """

---

## 🎯 New Organized Structure

### 🔧 Executable Scripts (in bin/)
```bash
# Convenience commands
./bin/pill-cli              # Launch CLI interface
./bin/pill-web              # Launch Web UI
./bin/pill-setup            # Setup environment  
./bin/pill-train            # Train model
./bin/pill-test             # Run tests

# Traditional tools  
./bin/setup                 # System setup
./bin/test                  # Testing utilities
./bin/train                 # Training scripts
./bin/deploy                # Deployment
./bin/monitor               # Monitoring
./bin/clean                 # Cleanup
```

### 📁 Clean Structure
- ✅ All executables in `bin/` directory
- ✅ Legacy files moved to `legacy/`
- ✅ No duplicate files
- ✅ Symlinks for backward compatibility

"""
            
            # Insert before "## 🤝 Contributing" section
            if "## 🤝 Contributing" in content:
                content = content.replace("## 🤝 Contributing", new_section + "## 🤝 Contributing")
            else:
                content += new_section
            
            readme_path.write_text(content)
            print(f"  📝 Updated: README.md")
    
    def create_final_summary(self):
        """Tạo báo cáo tóm tắt cuối cùng"""
        summary_file = self.project_root / "docs" / "CLEANUP_SUMMARY.md"
        
        summary_content = f"""# 🧹 Project Cleanup Summary

## 📊 Statistics

### 🗑️ Duplicates Removed
- **Files found**: {len(self.duplicates_found)}
- **Files deleted**: {len(self.files_deleted)}

### 📦 Files Moved
- **Total moved**: {len(self.files_moved)}

### 🔗 Symlinks Created  
- **Total created**: {len(self.links_created)}

## 📋 Detailed Report

### 🗑️ Deleted Duplicates
```
{chr(10).join(self.files_deleted)}
```

### 📦 File Moves
```
{chr(10).join([f"{src} → {dst}" for src, dst in self.files_moved])}
```

### 🔗 Symlinks Created
```
{chr(10).join([f"{link} → {target}" for link, target in self.links_created])}
```

## 🎯 New Project Structure

✅ **Organized executables** in `bin/` directory  
✅ **Legacy files** moved to `legacy/`  
✅ **No duplicate files** remaining  
✅ **Clean project root** with only essential files  
✅ **Convenience scripts** for easy access  
✅ **Backward compatibility** via symlinks  

## 🚀 Usage After Cleanup

```bash
# Quick access via convenience scripts
./bin/pill-cli              # CLI interface
./bin/pill-web              # Web interface  
./bin/pill-setup            # Environment setup
./bin/pill-train            # Model training
./bin/pill-test             # Testing

# Traditional commands still work
./run cli
./run web  
python main.py status
```

## ✨ Benefits

- 🎯 **Cleaner structure**: Easy to navigate
- ⚡ **Faster access**: All tools in one place
- 🔄 **Better organization**: Logical grouping
- 🧹 **No redundancy**: Duplicate files removed
- 📱 **User-friendly**: Convenience scripts added
- 🔗 **Compatible**: Old commands still work

---

*Cleanup completed on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        summary_file.write_text(summary_content)
        print(f"  📄 Created: CLEANUP_SUMMARY.md")
    
    def show_final_report(self):
        """Hiển thị báo cáo cuối cùng"""
        print("\n" + "=" * 80)
        print("🎉 PROJECT CLEANUP COMPLETED!")
        print("=" * 80)
        
        print(f"\n📊 Summary:")
        print(f"  🗑️ Duplicates deleted: {len(self.files_deleted)}")
        print(f"  📦 Files moved: {len(self.files_moved)}")
        print(f"  🔗 Symlinks created: {len(self.links_created)}")
        
        print(f"\n🎯 New structure highlights:")
        print(f"  ✅ All executables in bin/ directory")
        print(f"  ✅ Legacy files in legacy/ directory")
        print(f"  ✅ Clean project root")
        print(f"  ✅ Convenience scripts added")
        
        print(f"\n🚀 Quick start commands:")
        print(f"  ./bin/pill-cli              # CLI interface")
        print(f"  ./bin/pill-web              # Web interface")
        print(f"  ./bin/pill-setup            # Setup environment")
        
        print("\n✨ Project is now clean and organized!")
    
    def run_cleanup(self):
        """Chạy toàn bộ quá trình cleanup"""
        self.show_banner()
        
        try:
            # Step 1: Find and remove duplicates
            duplicates = self.find_duplicates()
            if duplicates:
                self.remove_duplicates()
            
            # Step 2: Organize executables
            self.organize_executables()
            
            # Step 3: Organize legacy files
            self.organize_legacy_files()
            
            # Step 4: Remove redundant directories
            self.remove_redundant_directories()
            
            # Step 5: Create launcher scripts
            self.create_launcher_scripts()
            
            # Step 6: Update documentation
            self.update_documentation()
            
            # Step 7: Create final summary
            self.create_final_summary()
            
            # Step 8: Show final report
            self.show_final_report()
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình cleanup: {e}")
            return False

def main():
    """Main function"""
    cleaner = AdvancedProjectCleaner()
    
    if cleaner.run_cleanup():
        print("\n🎉 Cleanup thành công!")
        return 0
    else:
        print("\n❌ Cleanup thất bại!")
        return 1

if __name__ == "__main__":
    exit(main())
