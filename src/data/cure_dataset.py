"""
CURE Dataset Loader for Smart Pill Recognition System
Handles loading and preprocessing of CURE pharmaceutical dataset
"""

import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

class CUREDataset(Dataset):
    """
    CURE Pharmaceutical Dataset Loader
    
    Dataset structure:
    - CURE_dataset_train_cut_bounding_box/
        - 0/ (pill class 0)
            - top/ (top view images)
            - bottom/ (bottom view images)
        - 1/ (pill class 1)
            - top/
            - bottom/
        ...
    - CURE_dataset_validation_cut_bounding_box/
    - CURE_dataset_test/
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        include_text: bool = True,
        image_size: int = 224
    ):
        """
        Initialize CURE Dataset
        
        Args:
            data_dir: Path to Dataset_BigData/CURE_dataset
            split: 'train', 'validation', or 'test'
            transform: Image transformations
            include_text: Whether to include text imprint data
            image_size: Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.include_text = include_text
        self.image_size = image_size
        
        # Set up dataset paths
        if split == "train":
            self.dataset_path = self.data_dir / "CURE_dataset_train_cut_bounding_box"
        elif split == "validation":
            self.dataset_path = self.data_dir / "CURE_dataset_validation_cut_bounding_box"
        elif split == "test":
            self.dataset_path = self.data_dir / "CURE_dataset_test"
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', or 'test'")
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load dataset
        self.samples = self._load_samples()
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logging.info(f"Loaded {len(self.samples)} samples from {split} split")
        logging.info(f"Found {len(self.classes)} classes: {self.classes}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset"""
        samples = []
        
        if self.split == "test":
            # Test dataset has different structure
            samples = self._load_test_samples()
        else:
            # Train/validation datasets
            samples = self._load_train_val_samples()
        
        return samples
    
    def _load_train_val_samples(self) -> List[Dict]:
        """Load samples from train/validation datasets"""
        samples = []
        
        # Iterate through class directories
        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Check for top and bottom subdirectories
            for view in ["top", "bottom"]:
                view_dir = class_dir / view
                if not view_dir.exists():
                    continue
                
                # Check if there's a Customer subdirectory (nested structure)
                customer_dir = view_dir / "Customer"
                search_dir = customer_dir if customer_dir.exists() else view_dir
                
                # Load all images from this view (support both .jpg and .png)
                for img_pattern in ["*.jpg", "*.png"]:
                    for img_file in search_dir.glob(img_pattern):
                        sample = {
                            "image_path": str(img_file),
                            "class_name": class_name,
                            "view": view,
                            "text_imprint": self._extract_text_from_filename(img_file.name),
                            "pill_id": f"{class_name}_{view}_{img_file.stem}"
                        }
                        samples.append(sample)
        
        return samples
    
    def _load_test_samples(self) -> List[Dict]:
        """Load samples from test dataset"""
        samples = []
        
        # Test dataset has reference images and test images
        for img_file in self.dataset_path.glob("*.jpg"):
            filename = img_file.name
            
            # Parse filename to extract information
            parts = filename.split("_")
            if len(parts) >= 3:
                class_name = parts[0]
                view = parts[1] if "ref" not in filename else "reference"
                
                sample = {
                    "image_path": str(img_file),
                    "class_name": class_name,
                    "view": view,
                    "text_imprint": self._extract_text_from_filename(filename),
                    "pill_id": img_file.stem
                }
                samples.append(sample)
        
        return samples
    
    def _extract_text_from_filename(self, filename: str) -> str:
        """Extract text imprint information from filename"""
        # This is a placeholder - you may need to implement
        # actual text extraction based on your dataset
        if "ref" in filename:
            return "REFERENCE"
        
        # Extract numbers or patterns from filename
        parts = filename.replace(".jpg", "").split("_")
        text_parts = [part for part in parts if part.isdigit()]
        
        return "_".join(text_parts) if text_parts else "UNKNOWN"
    
    def _get_classes(self) -> List[str]:
        """Get list of all classes in the dataset"""
        classes = set()
        for sample in self.samples:
            classes.add(sample["class_name"])
        return sorted(list(classes))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Prepare return data
        data = {
            "image": image,
            "class_name": sample["class_name"],
            "class_idx": self.class_to_idx[sample["class_name"]],
            "view": sample["view"],
            "pill_id": sample["pill_id"]
        }
        
        # Add text if requested
        if self.include_text:
            data["text_imprint"] = sample["text_imprint"]
        
        return data
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        distribution = {}
        for sample in self.samples:
            class_name = sample["class_name"]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def save_metadata(self, output_path: str):
        """Save dataset metadata to JSON file"""
        metadata = {
            "split": self.split,
            "num_samples": len(self.samples),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "class_distribution": self.get_class_distribution(),
            "dataset_path": str(self.dataset_path)
        }
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_path}")


def create_cure_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Args:
        dataset_root: Path to Dataset_BigData/CURE_dataset
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        image_size: Target image size
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Standard transform for validation and test
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CUREDataset(
        data_dir=dataset_root,
        split="train",
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = CUREDataset(
        data_dir=dataset_root,
        split="validation",
        transform=val_test_transform,
        image_size=image_size
    )
    
    test_dataset = CUREDataset(
        data_dir=dataset_root,
        split="test",
        transform=val_test_transform,
        image_size=image_size
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def analyze_cure_dataset(dataset_root: str):
    """
    Analyze CURE dataset and print statistics
    
    Args:
        dataset_root: Path to Dataset_BigData/CURE_dataset
    """
    print("🔍 Analyzing CURE Dataset...")
    print("=" * 50)
    
    # Create datasets
    datasets = {}
    for split in ["train", "validation", "test"]:
        try:
            dataset = CUREDataset(dataset_root, split=split)
            datasets[split] = dataset
            
            print(f"\n📊 {split.upper()} Dataset:")
            print(f"  • Samples: {len(dataset)}")
            print(f"  • Classes: {len(dataset.classes)}")
            
            # Class distribution
            distribution = dataset.get_class_distribution()
            print(f"  • Class distribution:")
            for class_name, count in sorted(distribution.items()):
                print(f"    - Class {class_name}: {count} samples")
            
            # Save metadata
            metadata_path = f"data/processed/{split}_metadata.json"
            os.makedirs("data/processed", exist_ok=True)
            dataset.save_metadata(metadata_path)
            
        except Exception as e:
            print(f"  ❌ Error loading {split} dataset: {e}")
    
    # Overall statistics
    total_samples = sum(len(ds) for ds in datasets.values())
    total_classes = len(set().union(*[ds.classes for ds in datasets.values()]))
    
    print(f"\n📈 Overall Statistics:")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Total classes: {total_classes}")
    print(f"  • Splits available: {list(datasets.keys())}")
    
    return datasets


if __name__ == "__main__":
    # Example usage
    dataset_root = "Dataset_BigData/CURE_dataset"
    
    # Analyze dataset
    datasets = analyze_cure_dataset(dataset_root)
    
    # Create data loaders
    print("\n🚀 Creating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = create_cure_dataloaders(
            dataset_root=dataset_root,
            batch_size=16,  # Smaller batch for testing
            num_workers=2,
            image_size=224
        )
        
        print(f"✅ DataLoaders created successfully!")
        print(f"  • Train batches: {len(train_loader)}")
        print(f"  • Validation batches: {len(val_loader)}")
        print(f"  • Test batches: {len(test_loader)}")
        
        # Test loading a batch
        print("\n🧪 Testing batch loading...")
        for batch in train_loader:
            print(f"  • Batch shape: {batch['image'].shape}")
            print(f"  • Classes in batch: {batch['class_name']}")
            break
            
    except Exception as e:
        print(f"❌ Error creating DataLoaders: {e}")
