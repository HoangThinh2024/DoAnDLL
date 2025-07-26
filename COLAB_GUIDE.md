# 🚀 Google Colab Guide - Smart Pill Recognition System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoangThinh2024/DoAnDLL/blob/main/Smart_Pill_Recognition_Colab.ipynb)

## 🌟 Overview

This guide shows you how to run the Smart Pill Recognition System in Google Colab with GPU acceleration and proper training functionality.

## 🎯 Quick Start

### Option 1: One-Click Notebook (Recommended)
1. Click the "Open in Colab" badge above
2. Run all cells in sequence
3. Upload your pill images and start recognizing!

### Option 2: Manual Setup
```python
# 1. Clone the repository
!git clone https://github.com/HoangThinh2024/DoAnDLL.git
%cd DoAnDLL

# 2. Run setup script
!python colab_setup.py

# 3. Start using the system
from colab_trainer import create_colab_trainer
trainer, model = create_colab_trainer()
```

## 🔧 Environment Setup

### Enable GPU Runtime
1. Go to `Runtime` → `Change runtime type`
2. Select `GPU` as Hardware accelerator
3. Choose `High-RAM` if available
4. Click `Save`

### Install Dependencies
```python
# The setup script will automatically install:
# - PyTorch with CUDA support
# - Transformers (Hugging Face)
# - Computer Vision libraries
# - Data processing tools
# - Visualization libraries

!python colab_setup.py
```

## 🏋️ Training Your Model

### Quick Training Example
```python
import sys
sys.path.append('/content/DoAnDLL')

from colab_trainer import ColabTrainer, ColabMultimodalPillTransformer
import torch

# Create model
model = ColabMultimodalPillTransformer(num_classes=1000)

# Create trainer
trainer = ColabTrainer(model, device='auto', mixed_precision=True)

# Training configuration
config = {
    'epochs': 10,
    'learning_rate': 2e-5,
    'save_path': '/content/checkpoints',
    'patience': 5
}

# Train the model (with your data)
# results = trainer.train(train_dataloader, val_dataloader, **config)
```

### Using Sample Data
```python
# The system includes sample data generation
from colab_trainer import create_colab_trainer

# This creates sample data and trains the model
trainer, model = create_colab_trainer(num_classes=10)

# Training results will be saved to /content/checkpoints
print("Model trained and saved!")
```

## 📊 Data Handling

### Upload Your Dataset
```python
from google.colab import files
import zipfile

# Upload dataset zip file
uploaded = files.upload()

# Extract dataset
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/data')
    print(f"Extracted {filename} to /content/data")
```

### Prepare Dataset for Training
```python
import os
from pathlib import Path

# Organize your data like this:
# /content/data/
# ├── train/
# │   ├── class1/
# │   ├── class2/
# │   └── ...
# └── val/
#     ├── class1/
#     ├── class2/
#     └── ...

dataset_path = "/content/data"
print(f"Dataset structure:")
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files
        print(f"{sub_indent}{file}")
    if len(files) > 3:
        print(f"{sub_indent}... and {len(files)-3} more files")
```

## 🔮 Inference and Testing

### Test with Uploaded Images
```python
from google.colab import files
from PIL import Image
import matplotlib.pyplot as plt

# Upload test image
uploaded = files.upload()

for filename in uploaded.keys():
    # Load and display image
    image = Image.open(filename)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Test Image: {filename}")
    plt.axis('off')
    plt.show()
    
    # Make prediction
    # result = model.predict(image, text_imprint="ADVIL 200")
    # print(f"Prediction: {result}")
```

### Batch Inference
```python
import os
from pathlib import Path

test_dir = "/content/test_images"
results = []

for image_path in Path(test_dir).glob("*.jpg"):
    # Load image
    image = Image.open(image_path)
    
    # Predict
    # prediction = model.predict(image)
    # results.append({
    #     'image': image_path.name,
    #     'prediction': prediction
    # })

print(f"Processed {len(results)} images")
```

## 📈 Monitoring and Visualization

### Training Progress
```python
import matplotlib.pyplot as plt

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage after training
# plot_training_history(results['history'])
```

### GPU Monitoring
```python
import torch

def check_gpu_status():
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory:")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB")
    else:
        print("❌ GPU not available")

check_gpu_status()
```

## 💾 Saving and Loading Models

### Save to Google Drive
```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Save model to Drive
import shutil

checkpoint_path = "/content/checkpoints/best_model.pth"
drive_path = "/content/drive/MyDrive/pill_recognition_model.pth"

if os.path.exists(checkpoint_path):
    shutil.copy2(checkpoint_path, drive_path)
    print(f"✅ Model saved to Google Drive: {drive_path}")
else:
    print("❌ No checkpoint found to save")
```

### Load from Google Drive
```python
# Load model from Drive
drive_path = "/content/drive/MyDrive/pill_recognition_model.pth"
local_path = "/content/checkpoints/loaded_model.pth"

if os.path.exists(drive_path):
    shutil.copy2(drive_path, local_path)
    
    # Load the model
    checkpoint = torch.load(local_path, map_location='cpu')
    print(f"✅ Model loaded from Google Drive")
    print(f"Model info: {checkpoint.keys()}")
else:
    print("❌ No model found in Google Drive")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. GPU Out of Memory
```python
# Reduce batch size
config['batch_size'] = 8  # Instead of 32

# Enable gradient checkpointing
config['gradient_checkpointing'] = True

# Clear GPU memory
torch.cuda.empty_cache()
```

#### 2. Dependencies Not Found
```python
# Reinstall dependencies
!pip install --upgrade torch torchvision transformers

# Restart runtime and run setup again
!python colab_setup.py
```

#### 3. Model Loading Issues
```python
# Check checkpoint contents
checkpoint = torch.load('path/to/model.pth', map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))

# Verify model architecture matches
print("Model architecture:", model)
```

#### 4. Training Stops Early
```python
# Increase patience
config['patience'] = 10  # Allow more epochs without improvement

# Reduce learning rate
config['learning_rate'] = 1e-5  # Smaller learning rate

# Check for NaN values
print("Checking for NaN in loss...")
```

### Performance Tips

1. **Use Mixed Precision**: Enable for faster training
   ```python
   trainer = ColabTrainer(model, mixed_precision=True)
   ```

2. **Optimize Batch Size**: Find the largest batch size that fits in memory
   ```python
   # Start with batch_size=8, increase gradually
   batch_sizes = [8, 16, 32, 64]
   ```

3. **Use DataLoader with Multiple Workers**:
   ```python
   dataloader = DataLoader(dataset, batch_size=32, num_workers=2)
   ```

4. **Save Checkpoints Regularly**:
   ```python
   config['save_every'] = 5  # Save every 5 epochs
   ```

## 📚 Additional Resources

- [Main Repository](https://github.com/HoangThinh2024/DoAnDLL)
- [Complete Documentation](README.md)
- [Training Guide](TRAINING_FIX_README.md)
- [Architecture Details](MULTI_METHOD_README.md)

## 🎉 Success! 

You're now ready to train and use the Smart Pill Recognition System in Google Colab! 

🔗 **Next Steps:**
1. Run the main notebook: `Smart_Pill_Recognition_Colab.ipynb`
2. Upload your pill dataset
3. Train your model
4. Test with real pill images
5. Save your results to Google Drive

Happy training! 🚀