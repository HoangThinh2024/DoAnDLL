#!/usr/bin/env python3
"""
Enhanced Multimodal CNN Training Script for Pill Recognition
Based on the problem statement requirements with comprehensive features

This script implements a sophisticated multimodal training pipeline that combines:
- RGB image features using ResNet-18
- Contour/shape features using Canny edge detection
- Texture features using Gabor filters
- Text features using BERT embeddings from OCR
- Advanced data augmentation with Albumentations
- Comprehensive evaluation metrics including mAP
- Enhanced training loop with improved early stopping

Author: DoAnDLL Team
Date: 2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import numpy as np
from PIL import Image
import multiprocessing
import json
import time
from datetime import datetime
import matplotlib.font_manager as fm

# --- TÍCH HỢP ALBUMENTATIONS ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

# OCR import with fallback
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    print("Warning: PaddleOCR not available. OCR features will be disabled.")
    OCR_AVAILABLE = False

# Configure multiprocessing for PyTorch
multiprocessing.set_start_method('spawn', force=True)

# Define paths and parameters
base_dir = "CURE_dataset_train_cut_bounding_box"
validation_dir = "CURE_dataset_validation_cut_bounding_box"
subdirs = ["bottom/Customer", "bottom/Reference", "top/Customer", "top/Reference"]
batch_size = 16
learning_rate = 1e-4
epochs_phase_2 = 30  # Tăng số epochs để có thể early stop
patience = 25  # Tăng patience để tránh dừng quá sớm - Số epochs chờ đợi để early stop (tăng từ 10 lên 25)
validation_split = 0.2  # Nếu không có tập validation riêng, dùng 20% từ tập train

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize OCR model (disable for training to avoid setup issues)
if OCR_AVAILABLE:
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print("✅ PaddleOCR initialized successfully")
    except Exception as e:
        print(f"Warning: PaddleOCR initialization failed: {e}")
        ocr = None
else:
    ocr = None  # Disable OCR if not available

# Initialize BERT tokenizer and model with error handling
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    print("✅ BERT model initialized successfully")
    BERT_AVAILABLE = True
except Exception as e:
    print(f"Warning: BERT initialization failed: {e}")
    print("Text features will be disabled")
    tokenizer = None
    bert_model = None
    BERT_AVAILABLE = False

# Define data augmentation transformations for training
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for validation (no augmentation)
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Enhanced Albumentations transforms with multi-target support
train_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    
    # Các phép biến đổi màu sắc chỉ áp dụng cho ảnh RGB
    A.OneOf([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.8),
    ], p=1.0), # p=1.0 để đảm bảo một trong hai được chọn

    # Thêm nhiễu
    A.GaussNoise(p=0.2),

    # *** "SPARK" AUGMENTATION (Coarse Dropout / Cutout) ***
    # Xóa một vài vùng trên ảnh để buộc model học các đặc trưng toàn cục hơn
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.5),

    # Chuẩn hóa và chuyển sang Tensor
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], additional_targets={'contour': 'image', 'texture': 'image'}) # Áp dụng cho cả contour và texture

# Biến đổi cho tập validation (chỉ resize, chuẩn hóa và chuyển sang tensor)
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], additional_targets={'contour': 'image', 'texture': 'image'})


# Define pre-trained CNN model
class CNNModel(nn.Module):
    def __init__(self, output_size=196):
        super(CNNModel, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')  # Use new weights parameter
        self.model.fc = nn.Linear(self.model.fc.in_features, output_size)

    def forward(self, x):
        return self.model(x)

# Define combined model for RGB, contour, texture, and imprinted text
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.rgb_model = CNNModel(output_size=128)
        self.contour_model = CNNModel(output_size=128)
        self.texture_model = CNNModel(output_size=128)
        self.text_model = nn.Linear(768, 128)  # 768: BERT embedding size
        self.fc = nn.Linear(128 * 4, 196)  # Combine all features for classification
        self.dropout = nn.Dropout(0.3)  # Thêm dropout để giảm overfitting

    def forward(self, rgb, contour, texture, text):
        rgb_features = self.rgb_model(rgb)
        contour_features = self.contour_model(contour)
        texture_features = self.texture_model(texture)
        text_features = self.text_model(text).squeeze(1)  # Squeeze the dimension for text features

        # Concatenate features
        combined_features = torch.cat((rgb_features, contour_features, texture_features, text_features), dim=1)
        combined_features = self.dropout(combined_features)  # Áp dụng dropout
        output = self.fc(combined_features)
        return output

# Backward compatibility alias
MultimodalCNNModel = CombinedModel

# Define text embedding function using BERT
def text_to_tensor(text):
    if not BERT_AVAILABLE:
        # Return zero tensor if BERT is not available
        return torch.zeros((1, 768), dtype=torch.float32)
    
    try:
        encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            bert_output = bert_model(**{key: val.to(device) for key, val in encoded_text.items()})
        return bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token's embedding
    except Exception as e:
        print(f"BERT encoding error: {e}")
        return torch.zeros((1, 768), dtype=torch.float32)

# Custom Dataset for pill images
class PillImageDataset(Dataset):
    def __init__(self, base_dir, subdirs, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for stt in range(196):  # STT ranges from 0 to 196
            for subdir in subdirs:
                dir_path = os.path.join(base_dir, str(stt), subdir)
                if not os.path.exists(dir_path):
                    continue
                for file_name in os.listdir(dir_path):
                    if file_name.endswith(".png") or file_name.endswith(".bmp"):
                        self.images.append(os.path.join(dir_path, file_name))
                        self.labels.append(stt)
        
        print(f"Loaded {len(self.images)} images from {base_dir}")
                        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                # Provide a placeholder image if reading fails
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Đảm bảo kích thước ảnh nhất quán
            image = cv2.resize(image, (224, 224))
            
            # 1. Trích xuất đặc trưng RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 2. Trích xuất đặc trưng hình dạng (shape/contour)
            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Áp dụng GaussianBlur để giảm nhiễu trước khi phát hiện cạnh
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Sử dụng Canny để phát hiện cạnh
            edges = cv2.Canny(blurred, 50, 150)
            
            # Tìm contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Tạo ảnh contour
            contour_image = np.zeros_like(image)
            cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
            
            # 3. Trích xuất đặc trưng texture
            # Phương pháp 1: Sử dụng bộ lọc Gabor để trích xuất texture
            # Khởi tạo kernel Gabor với các thông số khác nhau để bắt texture đa hướng
            texture_image = np.zeros_like(image)
            
            # Tạo bộ lọc Gabor với các góc khác nhau
            angles = [0, 45, 90, 135]
            for theta in angles:
                theta_rad = theta * np.pi / 180
                kernel = cv2.getGaborKernel((21, 21), 5.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                
                # Áp dụng bộ lọc
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                
                # Kết hợp kết quả
                texture_image[:,:,0] = np.maximum(texture_image[:,:,0], filtered)
                texture_image[:,:,1] = np.maximum(texture_image[:,:,1], filtered)
                texture_image[:,:,2] = np.maximum(texture_image[:,:,2], filtered)
            
            # Apply Albumentations transform if available
            if self.transform and hasattr(self.transform, 'additional_targets'):
                # Use Albumentations with multi-target support
                transformed = self.transform(image=rgb_image, contour=contour_image, texture=texture_image)
                rgb_tensor = transformed['image']
                contour_tensor = transformed['contour']
                texture_tensor = transformed['texture']
            else:
                # Use PIL/torchvision transforms (backward compatibility)
                pil_rgb = Image.fromarray(rgb_image)
                pil_contour = Image.fromarray(contour_image)
                pil_texture = Image.fromarray(texture_image)
                
                rgb_tensor = self.transform(pil_rgb) if self.transform else torch.tensor(rgb_image / 255.0, dtype=torch.float32).permute(2, 0, 1)
                contour_tensor = self.transform(pil_contour) if self.transform else torch.tensor(contour_image / 255.0, dtype=torch.float32).permute(2, 0, 1)
                texture_tensor = self.transform(pil_texture) if self.transform else torch.tensor(texture_image / 255.0, dtype=torch.float32).permute(2, 0, 1)
            
            # 4. Trích xuất text thông qua OCR
            try:
                if ocr is not None:
                    ocr_result = ocr.ocr(image_path, cls=True)
                    imprinted_text = ' '.join([line[1][0] for line in ocr_result[0]]) if ocr_result and ocr_result[0] else ""
                else:
                    imprinted_text = ""  # Skip OCR if disabled
            except Exception as e:
                print(f"OCR error on {image_path}: {e}")
                imprinted_text = ""
                
            text_tensor = text_to_tensor(imprinted_text)
            
            return rgb_tensor, contour_tensor, texture_tensor, text_tensor, label
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return default values if processing fails
            default_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            default_text = text_to_tensor("")
            return default_tensor, default_tensor, default_tensor, default_text, label

# Define custom loss function with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        return torch.mean(torch.sum(-weight * log_prob, dim=-1))

# Modified evaluation function to calculate mAP more efficiently
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []  # Store softmax probabilities
    
    with torch.no_grad():
        for rgb, contour, texture, text, labels in dataloader:
            rgb, contour, texture, text, labels = rgb.to(device), contour.to(device), texture.to(device), text.to(device), labels.to(device)
            
            outputs = model(rgb, contour, texture, text)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate softmax probabilities
            scores = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())  # No need for detach() here because we're using torch.no_grad()
    
    # Combine scores from all batches
    all_scores = np.vstack(all_scores)
    
    # Calculate basic metrics
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Calculate mAP
    try:
        # Convert labels to one-hot format
        num_classes = 196
        one_hot_labels = np.zeros((len(all_labels), num_classes))
        for i, label in enumerate(all_labels):
            if 0 <= label < num_classes:
                one_hot_labels[i, label] = 1
        
        # Ensure dimensions match
        if all_scores.shape[1] != num_classes:
            temp_scores = np.zeros((all_scores.shape[0], num_classes))
            min_classes = min(all_scores.shape[1], num_classes)
            temp_scores[:, :min_classes] = all_scores[:, :min_classes]
            all_scores = temp_scores
        
        # Calculate per-class mAP and weighted average
        mAP_per_class = []
        class_support = []
        
        for i in range(num_classes):
            if np.sum(one_hot_labels[:, i]) > 0:
                ap = average_precision_score(one_hot_labels[:, i], all_scores[:, i])
                support = np.sum(one_hot_labels[:, i])
                mAP_per_class.append(ap)
                class_support.append(support)
        
        mAP = np.average(mAP_per_class, weights=class_support) if mAP_per_class else 0.0
    except Exception as e:
        print(f"Error in mAP calculation: {e}")
        mAP = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP': mAP,
        'loss': val_loss / len(dataloader)
    }
    
    model.train()
    return metrics

# Training loop with validation and early stopping - Modified to save model with best mAP
def train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, epochs, patience, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    best_val_map = 0.0
    best_model_wts = None
    best_epoch = 0
    early_stop_counter = 0
    converged = False
    
    # Improved early stopping criteria
    min_improvement = 0.0001  # Giảm ngưỡng cải thiện tối thiểu từ 0.001 xuống 0.0001
    patience_buffer = 5  # Tăng buffer epochs từ 2 lên 5
    
    # Dictionaries to store metrics
    train_metrics_history = {
        'loss': [],
        'accuracy': [],
        'mAP': []  # Added mAP tracking for training set
    }
    
    val_metrics_history = {
        'loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mAP': []
    }
    
    # Learning rate history
    lr_history = []
    
    # Start timer for training
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        correct, total, epoch_loss = 0, 0, 0.0
        all_train_preds = []
        all_train_labels = []
        all_train_scores = []
        batch_times = []
        
        for batch_idx, (rgb, contour, texture, text, labels) in enumerate(train_loader):
            batch_start = time.time()
            
            rgb, contour, texture, text, labels = rgb.to(device), contour.to(device), texture.to(device), text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, contour, texture, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Collect data for mAP calculation
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            all_train_scores.append(scores.detach().cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # Print batch progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Time: {batch_time:.2f}s, "
                      f"Samples/sec: {len(labels)/batch_time:.1f}")
        
        # Calculate training metrics
        train_accuracy = correct / total
        train_loss = epoch_loss / len(train_loader)
        
        # Calculate training mAP
        try:
            all_train_scores = np.vstack(all_train_scores)
            # Convert labels to one-hot for mAP calculation
            num_classes = 196
            train_one_hot_labels = np.zeros((len(all_train_labels), num_classes))
            for i, label in enumerate(all_train_labels):
                if 0 <= label < num_classes:
                    train_one_hot_labels[i, label] = 1
            
            # Calculate per-class mAP and weighted average
            train_mAP_per_class = []
            train_class_support = []
            
            for i in range(num_classes):
                if np.sum(train_one_hot_labels[:, i]) > 0:
                    ap = average_precision_score(train_one_hot_labels[:, i], all_train_scores[:, i])
                    support = np.sum(train_one_hot_labels[:, i])
                    train_mAP_per_class.append(ap)
                    train_class_support.append(support)
            
            train_mAP = np.average(train_mAP_per_class, weights=train_class_support) if train_mAP_per_class else 0.0
        except Exception as e:
            print(f"Error calculating training mAP: {e}")
            train_mAP = 0.0
        
        # Store training metrics
        train_metrics_history['loss'].append(train_loss)
        train_metrics_history['accuracy'].append(train_accuracy)
        train_metrics_history['mAP'].append(train_mAP)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Store validation metrics
        for key, value in val_metrics.items():
            val_metrics_history[key].append(value)
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Update learning rate based on validation mAP
        scheduler.step(val_metrics['mAP'])
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}] completed in {epoch_time:.2f}s (avg batch: {avg_batch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train mAP: {train_mAP:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val mAP: {val_metrics['mAP']:.4f}")
        print(f"  Learning rate: {current_lr}")
        
        # Check if this is the best model by mAP (with improved criteria)
        improvement = val_metrics['mAP'] - best_val_map
        if improvement > min_improvement:
            best_val_map = val_metrics['mAP']
            best_model_wts = model.state_dict().copy()  # Ensure we make a copy
            best_epoch = epoch
            early_stop_counter = 0
            converged = True
            
            # Ensure save directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the best model with comprehensive checkpoint
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mAP': val_metrics['mAP'],
                'val_accuracy': val_metrics['accuracy'],
                'train_mAP': train_mAP,
                'train_accuracy': train_accuracy,
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'best_val_mAP': best_val_map,
                'improvement': improvement,
                'model_architecture': str(model),
                'training_config': {
                    'epochs': epochs,
                    'patience': patience,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
            }
            
            try:
                torch.save(checkpoint_data, best_model_path)
                print(f"  ✅ Saved new best model with mAP: {best_val_map:.4f} (improvement: +{improvement:.4f})")
                
                # Verify the saved model can be loaded
                test_load = torch.load(best_model_path, map_location='cpu')
                if 'model_state_dict' in test_load:
                    print(f"  ✅ Model checkpoint verified successfully")
                else:
                    print(f"  ⚠️  Warning: Model checkpoint missing model_state_dict")
                    
            except Exception as e:
                print(f"  ❌ Error saving model checkpoint: {e}")
                # Try to save to backup location
                backup_path = os.path.join(save_dir, f'best_model_backup_epoch_{epoch+1}.pth')
                torch.save(checkpoint_data, backup_path)
                print(f"  💾 Saved backup checkpoint to: {backup_path}")
        else:
            early_stop_counter += 1
            print(f"  No significant improvement (change: {improvement:.6f}, threshold: {min_improvement:.6f})")
            print(f"  Early stop counter: {early_stop_counter}/{patience + patience_buffer}")
        
        # Save checkpoint at regular intervals with improved error handling
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_metrics_history,
                'val_history': val_metrics_history,
                'current_val_mAP': val_metrics['mAP'],
                'current_val_accuracy': val_metrics['accuracy'],
                'best_val_mAP_so_far': best_val_map,
                'early_stop_counter': early_stop_counter
            }
            
            try:
                torch.save(checkpoint_data, checkpoint_path)
                print(f"  ✅ Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
                
                # Verify checkpoint integrity
                test_load = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' not in test_load:
                    print(f"  ⚠️  Warning: Checkpoint missing model_state_dict!")
                    
            except Exception as e:
                print(f"  ❌ Error saving checkpoint: {e}")
                # Continue training even if checkpoint fails
        
        # Save epoch-wise metrics to text file
        with open(os.path.join(save_dir, f'epoch_{epoch+1}_metrics.txt'), 'w') as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Training Loss: {train_loss:.6f}\n")
            f.write(f"Training Accuracy: {train_accuracy:.6f}\n")
            f.write(f"Training mAP: {train_mAP:.6f}\n")
            f.write(f"Validation Loss: {val_metrics['loss']:.6f}\n")
            f.write(f"Validation Accuracy: {val_metrics['accuracy']:.6f}\n")
            f.write(f"Validation mAP: {val_metrics['mAP']:.6f}\n")
            f.write(f"Validation Precision: {val_metrics['precision']:.6f}\n")
            f.write(f"Validation Recall: {val_metrics['recall']:.6f}\n")
            f.write(f"Validation F1: {val_metrics['f1']:.6f}\n")
            f.write(f"Learning Rate: {current_lr}\n")
            f.write(f"Epoch Time: {epoch_time:.2f} seconds\n")
        
        # Check for early stopping with improved criteria
        effective_patience = patience + patience_buffer
        
        # OPTION 1: Tắt early stopping để chạy đủ 30 epochs (DISABLED by default - uncomment to disable early stopping)
        # pass  # Tắt early stopping hoàn toàn - model sẽ chạy đủ số epochs được cấu hình
        
        # OPTION 2: Giữ early stopping nhưng với patience cao hơn (ACTIVE)
        if early_stop_counter >= effective_patience:
            print(f"🛑 Early stopping triggered after epoch {epoch+1}")
            print(f"   No improvement for {early_stop_counter} epochs (patience: {effective_patience})")
            print(f"   Best mAP achieved: {best_val_map:.4f} at epoch {best_epoch+1}")
            break
        elif early_stop_counter >= patience:
            print(f"⚠️  Warning: {early_stop_counter}/{effective_patience} patience epochs reached")
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"⏱️  Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"🏆 Best validation mAP: {best_val_map:.4f} achieved at epoch {best_epoch+1}")
    
    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model, {
        'train': train_metrics_history,
        'val': val_metrics_history,
        'best_epoch': best_epoch,
        'best_mAP': best_val_map,
        'training_time': total_time
    }

# Simple data loader creation function for demo
def create_simple_data_loaders():
    """Create simple data loaders for testing purposes when no real data is available"""
    print("⚠️  Creating demo data loaders (no real dataset found)")
    
    # Simple dummy dataset for testing
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Generate dummy data
            rgb = torch.randn(3, 224, 224)
            contour = torch.randn(3, 224, 224)
            texture = torch.randn(3, 224, 224)
            text = torch.randn(768)
            label = idx % 196  # Random label
            return rgb, contour, texture, text, label
    
    train_dataset = DummyDataset(500)
    val_dataset = DummyDataset(100)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Main training function
def main():
    """Main training function"""
    print("🚀 Starting Enhanced Multimodal CNN Training")
    print("=" * 60)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"enhanced_model_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {save_dir}")
    
    # Check if real dataset exists
    if os.path.exists(base_dir):
        print(f"✅ Found training dataset: {base_dir}")
        # Use real dataset implementation from original script
        from Dataset_BigData.CURE_dataset.train import create_data_loaders
        train_loader, val_loader = create_data_loaders(
            base_dir=base_dir,
            validation_dir=validation_dir,
            subdirs=subdirs,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=batch_size,
            validation_split=validation_split,
            use_separate_val=os.path.exists(validation_dir)
        )
    else:
        print(f"⚠️  Training dataset not found: {base_dir}")
        train_loader, val_loader = create_simple_data_loaders()
    
    # Initialize model
    print("🏗️  Initializing CombinedModel...")
    combined_model = CombinedModel().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    # Use Label Smoothing loss for better generalization
    criterion = LabelSmoothingCrossEntropy()
    print("📊 Using LabelSmoothingCrossEntropy loss")
    
    # Train model
    print("🏋️  Starting training...")
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=combined_model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=epochs_phase_2,
        patience=patience,
        save_dir=save_dir
    )
    
    print(f"✅ Training completed. Results saved in {save_dir}")
    
    # Save final summary
    summary = {
        'model_type': 'Enhanced Multimodal CNN',
        'features': ['RGB', 'Contour', 'Texture', 'Text'],
        'augmentation': 'Albumentations with multi-target support',
        'training_config': {
            'epochs': epochs_phase_2,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'patience': patience
        },
        'best_results': {
            'epoch': history['best_epoch'],
            'mAP': history['best_mAP']
        },
        'total_training_time': history['training_time']
    }
    
    with open(os.path.join(save_dir, 'final_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return model, history

if __name__ == '__main__':
    try:
        model, history = main()
        print("🎉 Enhanced multimodal training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()