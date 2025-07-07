#!/usr/bin/env python3
"""
Multimodal Pill Recognition Training Script for CURE Dataset
Optimized for Ubuntu 22.04 + NVIDIA Quadro 6000 + CUDA 12.8

This script trains a multimodal model that combines:
- RGB features (ResNet-18)
- Contour/Shape features (Canny edge detection)
- Texture features (Gabor filters)
- Text features (BERT + PaddleOCR)

Author: DoAnDLL Project
Date: 2024
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import multiprocessing

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import numpy as np
from PIL import Image
import json
import time

# Import our custom modules
try:
    from data.cure_dataset import CUREDataset
    from utils.port_manager import PortManager
    from utils.utils import set_seed
    IMPORTS_AVAILABLE = True
except ImportError:
    print("⚠️  Note: Some custom modules not available. Using fallback implementations.")
    IMPORTS_AVAILABLE = False

# Fallback implementations
if not IMPORTS_AVAILABLE:
    def set_seed(seed):
        """Fallback implementation for set_seed"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
multiprocessing.set_start_method('spawn', force=True)

# Import OCR and BERT
try:
    from paddleocr import PaddleOCR
    from transformers import BertTokenizer, BertModel
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some dependencies not available: {e}")
    print("Please run: ./setup to install all dependencies")
    DEPENDENCIES_AVAILABLE = False

class MultimodalPillModel(nn.Module):
    """Multimodal model combining RGB, contour, texture, and text features"""
    
    def __init__(self, num_classes=196, dropout_rate=0.3):
        super(MultimodalPillModel, self).__init__()
        
        # Individual feature extractors
        self.rgb_model = self._create_resnet_branch(output_size=128)
        self.contour_model = self._create_resnet_branch(output_size=128)
        self.texture_model = self._create_resnet_branch(output_size=128)
        
        # Text feature extractor (BERT -> Linear)
        self.text_model = nn.Linear(768, 128)  # BERT embedding size
        
        # Fusion and classification layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def _create_resnet_branch(self, output_size=128):
        """Create a ResNet-18 branch for feature extraction"""
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output_size)
        return model
    
    def forward(self, rgb, contour, texture, text):
        # Extract features from each modality
        rgb_features = self.rgb_model(rgb)
        contour_features = self.contour_model(contour)
        texture_features = self.texture_model(texture)
        text_features = self.text_model(text).squeeze(1)
        
        # Concatenate all features
        combined_features = torch.cat((rgb_features, contour_features, texture_features, text_features), dim=1)
        
        # Fusion and classification
        fused_features = self.fusion_layer(combined_features)
        output = self.classifier(fused_features)
        
        return output

class CUREPillDataset(Dataset):
    """Dataset class for CURE pill images with multimodal features"""
    
    def __init__(self, base_dir, transform=None, device='cuda'):
        self.base_dir = Path(base_dir)
        self.transform = transform
        self.device = device
        
        # Initialize OCR and BERT
        if DEPENDENCIES_AVAILABLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        else:
            self.ocr = None
            self.tokenizer = None
            self.bert_model = None
        
        # Load dataset using our CURE dataset loader if available
        if IMPORTS_AVAILABLE:
            self.cure_loader = CUREDataset(str(self.base_dir))
        else:
            self.cure_loader = None
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load all samples from the CURE dataset"""
        samples = []
        
        # Iterate through all STT folders (0-195 for 196 classes)
        for stt in range(196):
            stt_path = self.base_dir / str(stt)
            if not stt_path.exists():
                continue
                
            # Check different subdirectories
            for subdir in ["bottom/Customer", "bottom/Reference", "top/Customer", "top/Reference"]:
                subdir_path = stt_path / subdir
                if not subdir_path.exists():
                    continue
                    
                # Find all image files
                for img_file in subdir_path.glob("*"):
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                        samples.append((str(img_file), stt))
        
        print(f"Loaded {len(samples)} samples from CURE dataset")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        try:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            image = cv2.resize(image, (224, 224))
            
            # 1. RGB features
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            rgb_tensor = self.transform(pil_image) if self.transform else self._to_tensor(rgb_image)
            
            # 2. Contour features
            contour_tensor = self._extract_contour_features(image)
            
            # 3. Texture features
            texture_tensor = self._extract_texture_features(image)
            
            # 4. Text features
            text_tensor = self._extract_text_features(image_path)
            
            return rgb_tensor, contour_tensor, texture_tensor, text_tensor, label
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return default values
            default_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
            default_text = self._get_default_text_tensor()
            return default_tensor, default_tensor, default_tensor, default_text, label
    
    def _extract_contour_features(self, image):
        """Extract contour/shape features using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create contour image
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        
        contour_pil = Image.fromarray(contour_image)
        return self.transform(contour_pil) if self.transform else self._to_tensor(contour_image)
    
    def _extract_texture_features(self, image):
        """Extract texture features using Gabor filters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_image = np.zeros_like(image)
        
        # Apply Gabor filters at different angles
        for theta in [0, 45, 90, 135]:
            theta_rad = theta * np.pi / 180
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            # Combine results
            texture_image[:, :, 0] = np.maximum(texture_image[:, :, 0], filtered)
            texture_image[:, :, 1] = np.maximum(texture_image[:, :, 1], filtered)
            texture_image[:, :, 2] = np.maximum(texture_image[:, :, 2], filtered)
        
        texture_pil = Image.fromarray(texture_image)
        return self.transform(texture_pil) if self.transform else self._to_tensor(texture_image)
    
    def _extract_text_features(self, image_path):
        """Extract text features using OCR + BERT"""
        if not DEPENDENCIES_AVAILABLE or self.ocr is None:
            return self._get_default_text_tensor()
        
        try:
            ocr_result = self.ocr.ocr(image_path, cls=True)
            text = ' '.join([line[1][0] for line in ocr_result[0]]) if ocr_result and ocr_result[0] else ""
            
            # Convert text to BERT embedding
            encoded_text = self.tokenizer(text, padding='max_length', truncation=True, 
                                        max_length=128, return_tensors='pt')
            
            with torch.no_grad():
                bert_output = self.bert_model(**{key: val.to(self.device) for key, val in encoded_text.items()})
            
            return bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        except Exception as e:
            print(f"OCR error on {image_path}: {e}")
            return self._get_default_text_tensor()
    
    def _to_tensor(self, image):
        """Convert numpy image to tensor"""
        return torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
    
    def _get_default_text_tensor(self):
        """Get default text tensor when OCR fails"""
        return torch.zeros((1, 768), dtype=torch.float32)

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        return torch.mean(torch.sum(-weight * log_prob, dim=-1))

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for rgb, contour, texture, text, labels in dataloader:
            rgb = rgb.to(device)
            contour = contour.to(device)
            texture = texture.to(device)
            text = text.to(device)
            labels = labels.to(device)
            
            outputs = model(rgb, contour, texture, text)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = val_loss / len(dataloader)
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate mAP
    try:
        all_labels_array = np.array(all_labels)
        all_scores_array = np.array(all_scores)
        
        # One-hot encode labels
        num_classes = all_scores_array.shape[1]
        one_hot_labels = np.eye(num_classes)[all_labels_array]
        
        # Calculate AP for each class
        mAP_per_class = []
        class_support = []
        
        for i in range(num_classes):
            if np.sum(one_hot_labels[:, i]) > 0:
                ap = average_precision_score(one_hot_labels[:, i], all_scores_array[:, i])
                support = np.sum(one_hot_labels[:, i])
                mAP_per_class.append(ap)
                class_support.append(support)
        
        mAP = np.average(mAP_per_class, weights=class_support) if mAP_per_class else 0.0
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        mAP = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP': mAP
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_dir, patience=5):
    """Train the multimodal model"""
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Results will be saved to: {save_dir}")
    
    best_val_mAP = 0.0
    best_model_state = None
    early_stop_counter = 0
    
    train_history = {'loss': [], 'accuracy': [], 'mAP': []}
    val_history = {'loss': [], 'accuracy': [], 'mAP': [], 'precision': [], 'recall': [], 'f1': []}
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\\nEpoch [{epoch+1}/{num_epochs}]")
        
        for batch_idx, (rgb, contour, texture, text, labels) in enumerate(train_loader):
            rgb = rgb.to(device)
            contour = contour.to(device)
            texture = texture.to(device)
            text = text.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, contour, texture, text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")
        
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['mAP'])
        
        # Record metrics
        train_history['loss'].append(avg_train_loss)
        train_history['accuracy'].append(train_accuracy)
        
        for key, value in val_metrics.items():
            val_history[key].append(value)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"  Epoch completed in {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val mAP: {val_metrics['mAP']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['mAP'] > best_val_mAP:
            best_val_mAP = val_metrics['mAP']
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_mAP': val_metrics['mAP'],
                'val_accuracy': val_metrics['accuracy'],
                'train_history': train_history,
                'val_history': val_history
            }, save_dir / 'best_model.pth')
            
            print(f"  ✅ New best model saved! mAP: {best_val_mAP:.4f}")
        else:
            early_stop_counter += 1
            print(f"  No improvement. Early stop counter: {early_stop_counter}/{patience}")
        
        # Early stopping
        if early_stop_counter >= patience:
            print(f"\\n🛑 Early stopping triggered after epoch {epoch+1}")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {'train': train_history, 'val': val_history}

def create_data_loaders(dataset_path, batch_size=16, validation_split=0.2, num_workers=0):
    """Create training and validation data loaders"""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create full dataset
    full_dataset = CUREPillDataset(dataset_path, transform=train_transform)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Dataset split: {train_size} training, {val_size} validation samples")
    
    return train_loader, val_loader

def save_training_plots(history, save_dir):
    """Save training plots"""
    train_history = history['train']
    val_history = history['val']
    
    # Loss plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_history['loss'], label='Training Loss')
    plt.plot(val_history['loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_history['accuracy'], label='Training Accuracy')
    plt.plot(val_history['accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # mAP plot
    plt.subplot(1, 3, 3)
    if 'mAP' in train_history:
        plt.plot(train_history['mAP'], label='Training mAP')
    plt.plot(val_history['mAP'], label='Validation mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to {save_dir / 'training_plots.png'}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Multimodal Pill Recognition Model on CURE Dataset')
    parser.add_argument('--dataset-path', type=str, default='Dataset_BigData/CURE_dataset/CURE_dataset_train_cut_bounding_box',
                       help='Path to CURE dataset training directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check if dependencies are available
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Required dependencies not available. Please run ./setup first.")
        return
    
    # Check if dataset exists
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please ensure the CURE dataset is available in Dataset_BigData/CURE_dataset/")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"training_results_{timestamp}")
    save_dir.mkdir(exist_ok=True)
    
    print(f"📁 Results will be saved to: {save_dir}")
    
    # Save training configuration
    config = {
        'dataset_path': str(dataset_path),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'patience': args.patience,
        'seed': args.seed,
        'device': str(device),
        'timestamp': timestamp
    }
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Create data loaders
        print("\\n📊 Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            dataset_path=dataset_path,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            num_workers=args.num_workers
        )
        
        # Initialize model
        print("\\n🧠 Initializing multimodal model...")
        model = MultimodalPillModel(num_classes=196).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Setup training components
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        
        # Setup loss function with class weights
        try:
            print("\\n⚖️  Calculating class weights...")
            # Collect labels for class weight calculation
            all_labels = []
            for _, _, _, _, labels in train_loader:
                all_labels.extend(labels.numpy())
            
            class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            print("✅ Using weighted CrossEntropyLoss for balanced training")
            
        except Exception as e:
            print(f"⚠️  Error computing class weights: {e}")
            criterion = LabelSmoothingCrossEntropy()
            print("Using LabelSmoothingCrossEntropy loss instead")
        
        # Train model
        print("\\n🏋️  Starting training...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.epochs,
            device=device,
            save_dir=save_dir,
            patience=args.patience
        )
        
        # Save final model
        torch.save(model.state_dict(), save_dir / 'final_model.pth')
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {}
            for split in history:
                history_json[split] = {}
                for metric, values in history[split].items():
                    history_json[split][metric] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
            json.dump(history_json, f, indent=2)
        
        # Create training plots
        save_training_plots(history, save_dir)
        
        # Final evaluation
        print("\\n📈 Final evaluation...")
        final_metrics = evaluate_model(model, val_loader, criterion, device)
        
        print("\\n🎉 Training completed successfully!")
        print("📊 Final Results:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\\n💾 All results saved to: {save_dir}")
        print("\\nNext steps:")
        print("1. Check training plots and metrics")
        print("2. Test the model using: python recognition.py")
        print("3. Deploy using: ./deploy")
        
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
