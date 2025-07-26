#!/usr/bin/env python3
"""
Web Training Module for Smart Pill Recognition
Dedicated training script for Web UI integration
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import threading
import signal

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class WebTrainingManager:
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.training_process = None
        self.training_active = False
        self.training_log_file = None
        
    def start_training(self, epochs=30, batch_size=16, learning_rate=1e-4, save_dir=None):
        """Start real training process for Web UI"""
        # Check if training is actually active or just a stale state
        if self.training_active:
            # Verify if process is really running
            if self.training_process is not None:
                poll = self.training_process.poll()
                if poll is not None:
                    # Process has finished, reset state
                    self.training_active = False
                    self.training_process = None
                else:
                    # Process is still running
                    return {"status": "error", "message": "Training already active"}
            else:
                # No process but marked as active - reset state
                self.training_active = False
            
        # Setup save directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = self.project_root / "checkpoints" / f"web_training_{timestamp}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training command
        train_script = self.project_root / "train.py"
        if not train_script.exists():
            # Look for train.py in Dataset_BigData/CURE_dataset/
            train_script = self.project_root / "Dataset_BigData" / "CURE_dataset" / "train.py"
            
        if not train_script.exists():
            return {"status": "error", "message": f"Training script not found: {train_script}"}
        
        # Create temporary config for training
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "save_dir": str(save_dir),
            "patience": 25,  # Large patience to avoid early stopping
            "min_improvement": 0.00001
        }
        
        config_file = save_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Setup log file
        self.training_log_file = save_dir / "training.log"
        
        # Create enhanced training command
        cmd = [
            sys.executable, str(train_script),
            f"--epochs={epochs}",
            f"--batch_size={batch_size}",
            f"--learning_rate={learning_rate}",
            f"--save_dir={save_dir}",
            "--disable_early_stopping"  # Force disable early stopping
        ]
        
        try:
            # Change to the directory containing train.py
            work_dir = train_script.parent
            
            # Start training process in background
            with open(self.training_log_file, 'w') as log_file:
                self.training_process = subprocess.Popen(
                    cmd,
                    cwd=work_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
            
            self.training_active = True
            
            return {
                "status": "success", 
                "message": f"Training started with PID {self.training_process.pid}",
                "save_dir": str(save_dir),
                "log_file": str(self.training_log_file),
                "config": config
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to start training: {str(e)}"}
    
    def get_training_status(self):
        """Get current training status and progress"""
        if not self.training_active or self.training_process is None:
            return {"status": "inactive", "progress": 0}
            
        # Check if process is still running
        poll = self.training_process.poll()
        if poll is not None:
            self.training_active = False
            return {
                "status": "completed" if poll == 0 else "failed",
                "exit_code": poll,
                "progress": 100 if poll == 0 else 0
            }
        
        # Parse log file for progress
        progress_info = self._parse_training_log()
        return {
            "status": "running",
            "pid": self.training_process.pid,
            **progress_info
        }
    
    def _parse_training_log(self):
        """Parse training log file for progress information"""
        if not self.training_log_file or not self.training_log_file.exists():
            return {"progress": 0, "current_epoch": 0, "total_epochs": 0}
            
        try:
            with open(self.training_log_file, 'r') as f:
                lines = f.readlines()
                
            current_epoch = 0
            total_epochs = 0
            latest_metrics = {}
            
            for line in lines:
                # Look for epoch information
                if "Epoch [" in line and "/" in line:
                    try:
                        # Extract epoch info like "Epoch [5/30]"
                        epoch_part = line.split("Epoch [")[1].split("]")[0]
                        current, total = map(int, epoch_part.split("/"))
                        current_epoch = current
                        total_epochs = total
                    except:
                        pass
                        
                # Look for metrics
                if "Val mAP:" in line:
                    try:
                        parts = line.split("Val mAP:")
                        if len(parts) > 1:
                            map_value = float(parts[1].strip().split()[0])
                            latest_metrics["val_mAP"] = map_value
                    except:
                        pass
                        
                if "Val Accuracy:" in line:
                    try:
                        parts = line.split("Val Accuracy:")
                        if len(parts) > 1:
                            acc_value = float(parts[1].strip().split()[0])
                            latest_metrics["val_accuracy"] = acc_value
                    except:
                        pass
            
            progress = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
            
            return {
                "progress": min(progress, 100),
                "current_epoch": current_epoch,
                "total_epochs": total_epochs,
                "metrics": latest_metrics
            }
            
        except Exception as e:
            return {"progress": 0, "current_epoch": 0, "total_epochs": 0, "error": str(e)}
    
    def stop_training(self):
        """Stop current training process"""
        if not self.training_active or self.training_process is None:
            return {"status": "error", "message": "No active training to stop"}
            
        try:
            # Send SIGTERM first
            self.training_process.terminate()
            
            # Wait a bit
            try:
                self.training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't respond
                self.training_process.kill()
                self.training_process.wait()
            
            self.training_active = False
            return {"status": "success", "message": "Training stopped"}
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to stop training: {str(e)}"}
    
    def get_training_log(self, lines=50):
        """Get recent lines from training log"""
        if not self.training_log_file or not self.training_log_file.exists():
            return ""
            
        try:
            with open(self.training_log_file, 'r') as f:
                all_lines = f.readlines()
                
            # Return last N lines
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            return ''.join(recent_lines)
            
        except Exception as e:
            return f"Error reading log: {str(e)}"

# Global instance for Web UI
web_training_manager = WebTrainingManager()

def start_web_training(**kwargs):
    """Wrapper function for Web UI"""
    return web_training_manager.start_training(**kwargs)

def get_web_training_status():
    """Wrapper function for Web UI"""
    return web_training_manager.get_training_status()

def stop_web_training():
    """Wrapper function for Web UI"""
    return web_training_manager.stop_training()

def get_web_training_log(lines=50):
    """Wrapper function for Web UI"""
    return web_training_manager.get_training_log(lines)
