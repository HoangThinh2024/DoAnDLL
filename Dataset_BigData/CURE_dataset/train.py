import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
from paddleocr import PaddleOCR
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
#nohup python3 your_script.py > /dev/null 2>&1 &
# nohup python main_Basic_model.py > /dev/null 2>&1 
font_path = os.path.expanduser('venv/fonts/TIMES.TTF')
fm.fontManager.addfont(font_path)


multiprocessing.set_start_method('spawn', force=True)

# Define paths and parameters
base_dir = "CURE_dataset_train_cut_bounding_box"
validation_dir = "CURE_dataset_validation_cut_bounding_box"  # Thêm đường dẫn tới tập validation
subdirs = ["bottom/Customer", "bottom/Reference", "top/Customer", "top/Reference"]
batch_size = 16
learning_rate = 1e-4
epochs_phase_2 = 30  # Tăng số epochs để có thể early stop
patience = 5  # Số epochs chờ đợi để early stop
validation_split = 0.2  # Nếu không có tập validation riêng, dùng 20% từ tập train

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize OCR model
#ocr = PaddleOCR(use_angle_cls=True, lang='en')
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

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


train_transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
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
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, 
                    min_holes=1, min_height=8, min_width=8, p=0.5),

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
        self.model = models.resnet18(pretrained=True)
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

# Define text embedding function using BERT
def text_to_tensor(text):
    encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        bert_output = bert_model(**{key: val.to(device) for key, val in encoded_text.items()})
    return bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token's embedding

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
            pil_image = Image.fromarray(rgb_image)
            rgb_tensor = self.transform(pil_image) if self.transform else torch.tensor(rgb_image / 255.0, dtype=torch.float32).permute(2, 0, 1)
            
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
            
            # Chuyển đổi sang tensor
            contour_pil = Image.fromarray(contour_image)
            contour_tensor = self.transform(contour_pil) if self.transform else torch.tensor(contour_image / 255.0, dtype=torch.float32).permute(2, 0, 1)
            
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
            
            # Chuyển đổi sang tensor
            texture_pil = Image.fromarray(texture_image)
            texture_tensor = self.transform(texture_pil) if self.transform else torch.tensor(texture_image / 255.0, dtype=torch.float32).permute(2, 0, 1)
            
            # 4. Trích xuất text thông qua OCR
            try:
                ocr_result = ocr.ocr(image_path, cls=True)
                imprinted_text = ' '.join([line[1][0] for line in ocr_result[0]]) if ocr_result and ocr_result[0] else ""
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
        
        # Check if this is the best model by mAP
        if val_metrics['mAP'] > best_val_map:
            best_val_map = val_metrics['mAP']
            best_model_wts = model.state_dict()
            best_epoch = epoch
            early_stop_counter = 0
            converged = True
            
            # Save the best model
            torch.save({
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
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"  Saved new best model with mAP: {best_val_map:.4f}")
        else:
            early_stop_counter += 1
            print(f"  No improvement. Early stop counter: {early_stop_counter}/{patience}")
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_history': train_metrics_history,
                'val_history': val_metrics_history,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  Checkpoint saved at epoch {epoch+1}")
        
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
        
        # Check for early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Best validation mAP: {best_val_map:.4f} achieved at epoch {best_epoch+1}")
    
    # Save format summary
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write("===== Training Summary=====\n\n")
        
        # Dataset statistics
        f.write("Dataset Information:\n")
        f.write(f"- Training samples: {len(train_loader.dataset)}\n")
        f.write(f"- Validation samples: {len(val_loader.dataset)}\n")
        f.write(f"- Batch size: {train_loader.batch_size}\n\n")
        
        # Model architecture (simplified)
        f.write("Model Architecture:\n")
        f.write("- Combined model with ResNet-18 backbone for RGB, contour, and texture features\n")
        f.write("- BERT text embeddings for imprinted text\n")
        f.write("- 196 output classes\n\n")
        
        # Training parameters
        f.write("Training Parameters:\n")
        f.write(f"- Initial learning rate: {learning_rate}\n")
        f.write(f"- Optimizer: Adam with weight decay {optimizer.param_groups[0]['weight_decay']}\n")
        f.write(f"- Scheduler: ReduceLROnPlateau (factor={scheduler.factor}, patience={scheduler.patience})\n")
        f.write(f"- Early stopping patience: {patience}\n")
        f.write(f"- Total epochs configured: {epochs}\n")
        f.write(f"- Actual epochs run: {epoch+1}\n\n")
        
        # Convergence and early stopping
        f.write("Convergence Information:\n")
        if converged:
            f.write(f"- Model converged at epoch {best_epoch+1}\n")
        else:
            f.write("- Model did not converge within the configured epochs\n")
            
        if early_stop_counter >= patience:
            f.write(f"- Early stopping triggered at epoch {epoch+1}\n")
        else:
            f.write("- Early stopping not triggered\n\n")
        
        # Best performance
        f.write("Best Model Performance:\n")
        f.write(f"- Best validation mAP: {best_val_map:.6f} (Epoch {best_epoch+1})\n")
        if best_epoch < len(val_metrics_history['accuracy']):
            f.write(f"- Corresponding validation accuracy: {val_metrics_history['accuracy'][best_epoch]:.6f}\n")
            f.write(f"- Corresponding validation precision: {val_metrics_history['precision'][best_epoch]:.6f}\n")
            f.write(f"- Corresponding validation recall: {val_metrics_history['recall'][best_epoch]:.6f}\n")
            f.write(f"- Corresponding validation F1 score: {val_metrics_history['f1'][best_epoch]:.6f}\n")
        
        if best_epoch < len(train_metrics_history['accuracy']):
            f.write(f"- Corresponding training accuracy: {train_metrics_history['accuracy'][best_epoch]:.6f}\n")
            f.write(f"- Corresponding training mAP: {train_metrics_history['mAP'][best_epoch]:.6f}\n\n")
        
        # Training time
        f.write("Training Time:\n")
        f.write(f"- Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
        if epoch > 0:
            f.write(f"- Average time per epoch: {total_time / (epoch+1):.2f} seconds\n\n")
    
    # Plot and save metrics
    plot_and_save_metrics(train_metrics_history, val_metrics_history, save_dir)
    
    # Plot learning rate over epochs
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.plot(range(1, len(lr_history) + 1), lr_history, 'b-', marker='o', markevery=max(1, len(lr_history)//10))
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'learning_rate.pdf'), bbox_inches='tight')  # PDF for publications
    plt.close()
    
    # Create a combined dashboard for all metrics
    plt.figure(figsize=(16, 12))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    # Loss subplot
    plt.subplot(2, 2, 1)
    plt.plot(train_metrics_history['loss'], 'b-', marker='o', markevery=max(1, len(train_metrics_history['loss'])//10), label='Training')
    plt.plot(val_metrics_history['loss'], 'r--', marker='s', markevery=max(1, len(val_metrics_history['loss'])//10), label='Validation')
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy subplot
    plt.subplot(2, 2, 2)
    plt.plot(train_metrics_history['accuracy'], 'b-', marker='o', markevery=max(1, len(train_metrics_history['accuracy'])//10), label='Training')
    plt.plot(val_metrics_history['accuracy'], 'r--', marker='s', markevery=max(1, len(val_metrics_history['accuracy'])//10), label='Validation')
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # mAP subplot
    plt.subplot(2, 2, 3)
    plt.plot(train_metrics_history['mAP'], 'b-', marker='o', markevery=max(1, len(train_metrics_history['mAP'])//10), label='Training')
    plt.plot(val_metrics_history['mAP'], 'r--', marker='s', markevery=max(1, len(val_metrics_history['mAP'])//10), label='Validation')
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Precision/Recall/F1 subplot
    plt.subplot(2, 2, 4)
    plt.plot(val_metrics_history['precision'], 'g-', marker='o', markevery=max(1, len(val_metrics_history['precision'])//10), label='Precision')
    plt.plot(val_metrics_history['recall'], 'm--', marker='s', markevery=max(1, len(val_metrics_history['recall'])//10), label='Recall')
    plt.plot(val_metrics_history['f1'], 'c-.', marker='^', markevery=max(1, len(val_metrics_history['f1'])//10), label='F1 Score')
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Performance Metrics')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Combined Pill Classification Model Training Metrics', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
    plt.savefig(os.path.join(save_dir, 'combined_metrics_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'combined_metrics_dashboard.pdf'), bbox_inches='tight')  # PDF for publications
    plt.close()
    
    # Save training history as JSON
    history = {
        'train': train_metrics_history,
        'val': val_metrics_history,
        'best_epoch': best_epoch,
        'best_mAP': best_val_map,
        'training_time': total_time,
        'early_stopped': early_stop_counter >= patience,
        'converged': converged,
        'learning_rate_history': lr_history
    }
    
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Function to plot and save metrics
def plot_and_save_metrics(train_metrics, val_metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Set for all plots
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # preferred font
        'font.size': 10,                   # Base font size
        'axes.titlesize': 12,              # Title font size
        'axes.labelsize': 11,              # Axis label size
        'xtick.labelsize': 10,             # X-axis tick label size
        'ytick.labelsize': 10,             # Y-axis tick label size
        'legend.fontsize': 10,             # Legend font size
        'figure.titlesize': 14,            # Figure title size
        'figure.dpi': 600,                 # High resolution for publication
        'savefig.dpi': 600,                # High resolution for saving
    })
    
    # Find best epoch based on validation mAP
    best_epoch = np.argmax(val_metrics['mAP']) if val_metrics['mAP'] else 0
    
    # Plot loss - Train and Validation on the same graph
    plt.figure(figsize=(8, 6))
    plt.plot(train_metrics['loss'], 'b-', marker='o', 
             markevery=max(1, len(train_metrics['loss'])//10), 
             linewidth=1.5, label='Training Loss')
    plt.plot(val_metrics['loss'], 'r--', marker='s', 
             markevery=max(1, len(val_metrics['loss'])//10), 
             linewidth=1.5, label='Validation Loss')
    
    # Mark the best epoch
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right', frameon=True, framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'loss_plot.pdf'), bbox_inches='tight')  # PDF for publications
    plt.close()
    
    # Plot accuracy - Train and Validation on the same graph
    plt.figure(figsize=(8, 6))
    plt.plot(train_metrics['accuracy'], 'b-', marker='o', 
             markevery=max(1, len(train_metrics['accuracy'])//10), 
             linewidth=1.5, label='Training Accuracy')
    plt.plot(val_metrics['accuracy'], 'r--', marker='s', 
             markevery=max(1, len(val_metrics['accuracy'])//10), 
             linewidth=1.5, label='Validation Accuracy')
    
    # Mark the best epoch
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right', frameon=True, framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.pdf'), bbox_inches='tight')  # PDF for publications
    plt.close()
    
    # Plot mAP - Train and Validation on the same graph
    plt.figure(figsize=(8, 6))
    if 'mAP' in train_metrics:
        plt.plot(train_metrics['mAP'], 'b-', marker='o', 
                 markevery=max(1, len(train_metrics['mAP'])//10), 
                 linewidth=1.5, label='Training mAP')
    plt.plot(val_metrics['mAP'], 'r--', marker='s', 
             markevery=max(1, len(val_metrics['mAP'])//10), 
             linewidth=1.5, label='Validation mAP')
    
    # Mark the best epoch
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    
    # Add annotation for best mAP value
    if val_metrics['mAP']:
        best_map = val_metrics['mAP'][best_epoch]
        plt.annotate(f'Best mAP: {best_map:.4f}',
                     xy=(best_epoch, best_map),
                     xytext=(best_epoch + 0.5, best_map),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.title('Training and Validation mAP')
    plt.legend(loc='lower right', frameon=True, framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'map_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'map_plot.pdf'), bbox_inches='tight')  # PDF for publications
    plt.close()
    
    # Plot F1, Precision, Recall for Validation on the same graph
    plt.figure(figsize=(8, 6))
    plt.plot(val_metrics['precision'], 'g-', marker='o', 
             markevery=max(1, len(val_metrics['precision'])//10), 
             linewidth=1.5, label='Precision')
    plt.plot(val_metrics['recall'], 'm--', marker='s', 
             markevery=max(1, len(val_metrics['recall'])//10), 
             linewidth=1.5, label='Recall')
    plt.plot(val_metrics['f1'], 'c-.', marker='^', 
             markevery=max(1, len(val_metrics['f1'])//10), 
             linewidth=1.5, label='F1 Score')
    
    # Mark the best epoch
    plt.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.7, label='Best Model')
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Performance Metrics')
    plt.legend(loc='lower right', frameon=True, framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'metrics_plot.pdf'), bbox_inches='tight')  # PDF for publications
    plt.close()
    
    # Reset matplotlib parameters to defaults
    plt.rcParams.update(plt.rcParamsDefault)

# Class TransformSubset for applying transforms to split datasets
class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        try:
            rgb, contour, texture, text, label = self.subset[idx]
            if self.transform:
                # Chuyển tensor sang numpy và sau đó thành PIL Image để áp dụng transform
                # Xử lý trường hợp rgb có thể là tensor hoặc numpy array
                if isinstance(rgb, torch.Tensor):
                    pil_rgb = Image.fromarray((rgb.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil_contour = Image.fromarray((contour.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil_texture = Image.fromarray((texture.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                else:
                    # Giả sử rgb đã là numpy array với kênh màu cuối cùng (H, W, C)
                    pil_rgb = Image.fromarray(rgb)
                    pil_contour = Image.fromarray(contour)
                    pil_texture = Image.fromarray(texture)
                
                rgb = self.transform(pil_rgb)
                contour = self.transform(pil_contour)
                texture = self.transform(pil_texture)
            
            return rgb, contour, texture, text, label
        except Exception as e:
            print(f"Error in TransformSubset.__getitem__: {e}")
            # Trả về giá trị mặc định trong trường hợp lỗi
            default_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            default_text = torch.zeros((768,), dtype=torch.float32)
            return default_tensor, default_tensor, default_tensor, default_text, 0
    
    def __len__(self):
        return len(self.subset)

# Function to create data loaders
def create_data_loaders(base_dir, validation_dir, subdirs, train_transform, val_transform, batch_size, validation_split, use_separate_val):
    if use_separate_val:
        print(f"Using separate validation directory: {validation_dir}")
        # Load training dataset
        train_dataset = PillImageDataset(base_dir, subdirs, transform=train_transform)
        # Sử dụng num_workers=0 để tránh vấn đề CUDA IPC
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Load validation dataset
        val_dataset = PillImageDataset(validation_dir, subdirs, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        print(f"No validation directory found. Splitting training data with ratio {validation_split}")
        # Load and split the dataset
        full_dataset = PillImageDataset(base_dir, subdirs, transform=None)
        
        # Calculate split sizes
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Split the dataset
        generator = torch.Generator().manual_seed(42)  # Đặt seed để có thể tái tạo
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        # Áp dụng transform cho từng subset
        train_dataset = TransformSubset(train_subset, train_transform)
        val_dataset = TransformSubset(val_subset, val_transform)
        
        # Sử dụng num_workers=0 để tránh vấn đề CUDA IPC
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

# Main script
if __name__ == '__main__':
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"model_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if validation directory exists
    use_separate_val = os.path.exists(validation_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        base_dir=base_dir,
        validation_dir=validation_dir,
        subdirs=subdirs,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        validation_split=validation_split,
        use_separate_val=use_separate_val
    )

    # Initialize model
    combined_model = CombinedModel().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    
    # Handle class weights safely
    try:
        # Collect all labels from dataset
        all_labels = []
        if use_separate_val:
            # If using direct dataset
            if hasattr(train_loader.dataset, 'labels'):
                all_labels = train_loader.dataset.labels
            else:
                # If no labels attribute, iterate through dataset
                print("Collecting labels from dataset...")
                # Create a temporary loader with larger batch size for faster collection
                temp_loader = DataLoader(train_loader.dataset, batch_size=64, shuffle=False, num_workers=0)
                for batch in temp_loader:
                    all_labels.extend(batch[4].numpy())  # batch[4] is the label
        else:
            # With TransformSubset, we can collect labels from original subset
            if hasattr(train_loader.dataset.subset.dataset, 'labels'):
                # Get indices from subset
                indices = train_loader.dataset.subset.indices
                all_labels = [train_loader.dataset.subset.dataset.labels[i] for i in indices]
            else:
                print("Collecting labels from transformed subset...")
                # Iterate through dataset to collect labels
                for i in range(len(train_loader.dataset)):
                    _, _, _, _, label = train_loader.dataset[i]
                    all_labels.append(label)
        
        # Check if we have enough labels
        if len(all_labels) == 0:
            raise ValueError("No labels collected for class weight calculation")
            
        # Convert to numpy array if needed
        if not isinstance(all_labels, np.ndarray):
            all_labels = np.array(all_labels)
            
        # Ensure it's a 1D array
        all_labels = all_labels.flatten()
        
        # Print label distribution information
        print(f"Label distribution: {Counter(all_labels)}")
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print("Using weighted CrossEntropyLoss for imbalanced classes")
    except Exception as e:
        print(f"Error computing class weights: {e}")
        # Fallback to label smoothing if class weights calculation fails
        criterion = LabelSmoothingCrossEntropy()
        print("Using LabelSmoothingCrossEntropy loss")

    # Train model with modified evaluate_model function
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
    
    print(f"Training completed. Results saved in {save_dir}")
