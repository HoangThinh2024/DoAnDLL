#!/usr/bin/env python3
"""
Multimodal Pill Recognition Inference Script
Optimized for Ubuntu 22.04 + NVIDIA Quadro 6000 + CUDA 12.8

This script performs inference using a trained multimodal model on CURE dataset images.
It supports both single image and batch processing modes.

Author: DoAnDLL Project
Date: 2024
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import time
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our model
from train_cure_model import MultimodalPillModel, CUREPillDataset

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import dependencies
try:
    from paddleocr import PaddleOCR
    from transformers import BertTokenizer, BertModel
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

class PillRecognizer:
    """Multimodal pill recognition system"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # Initialize model
        self.model = MultimodalPillModel(num_classes=196).to(self.device)
        self._load_model()
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize OCR and BERT if available
        if DEPENDENCIES_AVAILABLE:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        else:
            self.ocr = None
            self.tokenizer = None
            self.bert_model = None
            print("⚠️  OCR and text features disabled due to missing dependencies")
    
    def _load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            if 'val_mAP' in checkpoint:
                print(f"📊 Model validation mAP: {checkpoint['val_mAP']:.4f}")
        else:
            self.model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully")
        
        self.model.eval()
    
    def _extract_features(self, image_path):
        """Extract all multimodal features from an image"""
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.resize(image, (224, 224))
        
        # 1. RGB features
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_tensor = self.transform(Image.fromarray(rgb_image)).unsqueeze(0).to(self.device)
        
        # 2. Contour features
        contour_tensor = self._extract_contour_features(image).unsqueeze(0).to(self.device)
        
        # 3. Texture features
        texture_tensor = self._extract_texture_features(image).unsqueeze(0).to(self.device)
        
        # 4. Text features
        text_tensor = self._extract_text_features(image_path).to(self.device)
        
        return rgb_tensor, contour_tensor, texture_tensor, text_tensor
    
    def _extract_contour_features(self, image):
        """Extract contour features using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
        
        return self.transform(Image.fromarray(contour_image))
    
    def _extract_texture_features(self, image):
        """Extract texture features using Gabor filters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_image = np.zeros_like(image)
        
        for theta in [0, 45, 90, 135]:
            theta_rad = theta * np.pi / 180
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            texture_image[:, :, 0] = np.maximum(texture_image[:, :, 0], filtered)
            texture_image[:, :, 1] = np.maximum(texture_image[:, :, 1], filtered)
            texture_image[:, :, 2] = np.maximum(texture_image[:, :, 2], filtered)
        
        return self.transform(Image.fromarray(texture_image))
    
    def _extract_text_features(self, image_path):
        """Extract text features using OCR and BERT"""
        if not DEPENDENCIES_AVAILABLE or self.ocr is None:
            return torch.zeros((1, 768), dtype=torch.float32).to(self.device)
        
        try:
            ocr_result = self.ocr.ocr(str(image_path), cls=True)
            text = ' '.join([line[1][0] for line in ocr_result[0]]) if ocr_result and ocr_result[0] else ""
            
            encoded_text = self.tokenizer(text, padding='max_length', truncation=True, 
                                        max_length=128, return_tensors='pt')
            
            with torch.no_grad():
                bert_output = self.bert_model(**{key: val.to(self.device) for key, val in encoded_text.items()})
            
            return bert_output.last_hidden_state[:, 0, :]
        
        except Exception as e:
            print(f"⚠️  OCR error on {image_path}: {e}")
            return torch.zeros((1, 768), dtype=torch.float32).to(self.device)
    
    def predict_single(self, image_path, return_probabilities=False):
        """Predict the class of a single image"""
        start_time = time.time()
        
        try:
            # Extract features
            rgb, contour, texture, text = self._extract_features(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(rgb, contour, texture, text)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            inference_time = time.time() - start_time
            
            result = {
                'image_path': str(image_path),
                'predicted_class': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time
            }
            
            if return_probabilities:
                result['probabilities'] = probabilities[0].cpu().numpy()
            
            return result
        
        except Exception as e:
            return {
                'image_path': str(image_path),
                'predicted_class': -1,
                'confidence': 0.0,
                'inference_time': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, image_paths, batch_size=16):
        """Predict classes for a batch of images"""
        results = []
        
        print(f"🔍 Processing {len(image_paths)} images in batches of {batch_size}...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            print(f"  Batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
            
            for image_path in batch_paths:
                result = self.predict_single(image_path)
                results.append(result)
        
        return results
    
    def evaluate_on_test_set(self, test_dir, ground_truth_mapping=None):
        """Evaluate model on a test set"""
        test_path = Path(test_dir)
        if not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(test_path.glob(f"**/{ext}"))
        
        if not image_paths:
            print("❌ No images found in test directory")
            return None
        
        print(f"📊 Evaluating on {len(image_paths)} test images...")
        
        # Run predictions
        results = self.predict_batch(image_paths)
        
        # Calculate statistics
        successful_predictions = [r for r in results if 'error' not in r]
        failed_predictions = [r for r in results if 'error' in r]
        
        stats = {
            'total_images': len(image_paths),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(failed_predictions),
            'success_rate': len(successful_predictions) / len(image_paths),
            'average_confidence': np.mean([r['confidence'] for r in successful_predictions]) if successful_predictions else 0,
            'average_inference_time': np.mean([r['inference_time'] for r in successful_predictions]) if successful_predictions else 0
        }
        
        # If ground truth mapping is provided, calculate accuracy
        if ground_truth_mapping:
            # Implementation would depend on how ground truth is structured
            pass
        
        return {
            'results': results,
            'statistics': stats
        }

def find_latest_model(search_dir="."):
    """Find the latest trained model"""
    search_path = Path(search_dir)
    
    # Look for training results directories
    model_dirs = list(search_path.glob("training_results_*"))
    if not model_dirs:
        return None
    
    # Sort by timestamp in directory name
    model_dirs.sort(key=lambda x: x.name.split('_')[-1], reverse=True)
    
    # Look for best model in the latest directory
    latest_dir = model_dirs[0]
    best_model_path = latest_dir / "best_model.pth"
    
    if best_model_path.exists():
        return best_model_path
    
    # Fallback to final model
    final_model_path = latest_dir / "final_model.pth"
    if final_model_path.exists():
        return final_model_path
    
    return None

def save_results(results, output_path):
    """Save prediction results to various formats"""
    output_path = Path(output_path)
    
    # Create results DataFrame
    df_data = []
    for result in results['results']:
        row = {
            'image_path': result['image_path'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'inference_time': result['inference_time']
        }
        if 'error' in result:
            row['error'] = result['error']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"📄 Results saved to: {csv_path}")
    
    # Save as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Results saved to: {json_path}")
    
    # Print statistics
    stats = results['statistics']
    print(f"\\n📊 Evaluation Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Successful predictions: {stats['successful_predictions']}")
    print(f"  Failed predictions: {stats['failed_predictions']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Average confidence: {stats['average_confidence']:.4f}")
    print(f"  Average inference time: {stats['average_inference_time']:.4f}s")

def main():
    """Main recognition function"""
    parser = argparse.ArgumentParser(description='Multimodal Pill Recognition Inference')
    parser.add_argument('--model-path', type=str, help='Path to trained model file')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--test-dir', type=str, help='Path to test directory for batch evaluation')
    parser.add_argument('--output', type=str, help='Output file path for results')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        print("⚠️  Some dependencies not available. Text features will be disabled.")
        print("For full functionality, run: ./setup")
    
    # Find model path
    model_path = args.model_path
    if not model_path:
        print("🔍 Looking for latest trained model...")
        model_path = find_latest_model()
        if not model_path:
            print("❌ No trained model found. Please specify --model-path or train a model first.")
            print("To train a model: ./train")
            return
        print(f"✅ Found model: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return
    
    # Initialize recognizer
    print(f"🧠 Initializing recognizer with device: {args.device}")
    try:
        recognizer = PillRecognizer(model_path, device=args.device)
    except Exception as e:
        print(f"❌ Failed to initialize recognizer: {e}")
        return
    
    # Single image prediction
    if args.image:
        print(f"🔍 Predicting single image: {args.image}")
        result = recognizer.predict_single(args.image, return_probabilities=True)
        
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
        else:
            print(f"\\n📊 Prediction Results:")
            print(f"  Image: {result['image_path']}")
            print(f"  Predicted Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Inference Time: {result['inference_time']:.4f}s")
            
            # Show top 5 predictions
            probs = result['probabilities']
            top5_indices = np.argsort(probs)[-5:][::-1]
            print(f"\\n  Top 5 Predictions:")
            for i, idx in enumerate(top5_indices, 1):
                print(f"    {i}. Class {idx}: {probs[idx]:.4f}")
    
    # Batch evaluation
    elif args.test_dir:
        print(f"📁 Evaluating test directory: {args.test_dir}")
        
        try:
            results = recognizer.evaluate_on_test_set(args.test_dir)
            
            if results:
                # Save results if output path specified
                if args.output:
                    save_results(results, args.output)
                else:
                    # Generate default output path
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"recognition_results_{timestamp}"
                    save_results(results, output_path)
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
    
    else:
        print("❌ Please specify either --image for single prediction or --test-dir for batch evaluation")
        print("Use --help for more information")

if __name__ == '__main__':
    main()
