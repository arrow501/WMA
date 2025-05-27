#!/usr/bin/env python3
"""
YOLO Training Script for Seat/Bin Detection
Supports training YOLOv5 models from scratch or pretrained weights
"""

import os
import shutil
import yaml
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import random
from ultralytics import YOLO
import torch

class YOLOTrainer:
    def __init__(self, data_dir=".", models_dir="Models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.photos_dir = self.data_dir / "Photos"
        self.labels_dir = self.data_dir / "Labels"
        self.labels_file = self.data_dir / "labels.txt"
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        # Load class names
        self.class_names = self._load_class_names()
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
    def _load_class_names(self):
        """Load class names from labels.txt"""
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        else:
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
    
    def prepare_dataset(self, train_split=0.8, val_split=0.2, random_state=42):
        """
        Prepare dataset by splitting into train/val sets with stratification
        """
        print("Preparing dataset...")
        
        # Get all image files
        image_files = []
        label_files = []
        
        for img_file in self.photos_dir.glob("*.jpg"):
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                image_files.append(img_file)
                label_files.append(label_file)
        
        print(f"Found {len(image_files)} image-label pairs")
        
        if len(image_files) == 0:
            raise ValueError("No matching image-label pairs found!")
        
        # Create stratification labels based on classes present in each image
        stratify_labels = []
        for label_file in label_files:
            classes_in_image = set()
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        classes_in_image.add(class_id)
            # Use first class as stratification key, or create composite key
            stratify_labels.append(min(classes_in_image) if classes_in_image else -1)
        
        # Split dataset
        train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
            image_files, label_files, 
            test_size=val_split, 
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Create dataset structure
        dataset_dir = self.data_dir / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        
        # Create directories
        train_img_dir = dataset_dir / "images" / "train"
        val_img_dir = dataset_dir / "images" / "val"
        train_lbl_dir = dataset_dir / "labels" / "train"
        val_lbl_dir = dataset_dir / "labels" / "val"
        
        for dir_path in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print("Copying training files...")
        for img, lbl in zip(train_imgs, train_lbls):
            shutil.copy2(img, train_img_dir / img.name)
            shutil.copy2(lbl, train_lbl_dir / lbl.name)
        
        print("Copying validation files...")
        for img, lbl in zip(val_imgs, val_lbls):
            shutil.copy2(img, val_img_dir / img.name)
            shutil.copy2(lbl, val_lbl_dir / lbl.name)
        
        # Create data.yaml
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"Dataset prepared:")
        print(f"  Train: {len(train_imgs)} images")
        print(f"  Val: {len(val_imgs)} images")
        print(f"  Data config: {yaml_path}")
        
        return yaml_path
    
    def train_model(self, model_size='s', pretrained=True, epochs=50, batch_size=8, 
                   img_size=640, device='cpu', patience=10):
        """
        Train YOLO model
        
        Args:
            model_size: 'n', 's', 'm', 'l', 'x'
            pretrained: Use pretrained weights
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: 'cpu' or 'cuda'
            patience: Early stopping patience
        """
        print(f"\nTraining YOLOv5{model_size} ({'pretrained' if pretrained else 'from scratch'})...")
        
        # Prepare dataset
        data_yaml_path = self.prepare_dataset()
        
        # Initialize model
        if pretrained:
            model_name = f"yolov5{model_size}.pt"
        else:
            model_name = f"yolov5{model_size}.yaml"
        
        model = YOLO(model_name)
        
        # Set device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device}")
        
        # Train
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            patience=patience,
            save=True,
            project=str(self.models_dir),
            name=f"yolov5{model_size}_{'pretrained' if pretrained else 'scratch'}",
            exist_ok=True
        )
        
        # Save model with descriptive name
        model_name = f"yolov5{model_size}_{'pretrained' if pretrained else 'scratch'}_best.pt"
        model_path = self.models_dir / model_name
        
        # Copy best model
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        if best_model_path.exists():
            shutil.copy2(best_model_path, model_path)
            print(f"Model saved to: {model_path}")
        
        return model_path, results
    
    def train_all_models(self, epochs=50, batch_size=8, device='cpu'):
        """Train all model variants"""
        model_sizes = ['n', 's', 'm']  # Start with smaller models for CPU
        pretrained_options = [True, False]
        
        results = {}
        
        for size in model_sizes:
            for pretrained in pretrained_options:
                try:
                    print(f"\n{'='*60}")
                    print(f"Training YOLOv5{size} ({'pretrained' if pretrained else 'scratch'})")
                    print(f"{'='*60}")
                    
                    model_path, train_results = self.train_model(
                        model_size=size,
                        pretrained=pretrained,
                        epochs=epochs,
                        batch_size=batch_size,
                        device=device
                    )
                    
                    results[f"yolov5{size}_{'pretrained' if pretrained else 'scratch'}"] = {
                        'model_path': model_path,
                        'results': train_results
                    }
                    
                except Exception as e:
                    print(f"Error training YOLOv5{size} ({'pretrained' if pretrained else 'scratch'}): {e}")
                    continue
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLO models for seat/bin detection')
    parser.add_argument('--model', '-m', default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (default: s)')
    parser.add_argument('--pretrained', '-p', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--device', '-d', default='cpu',
                       help='Device to use (default: cpu)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Train all model variants')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = YOLOTrainer()
    
    if args.all:
        print("Training all model variants...")
        results = trainer.train_all_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for model_name, result in results.items():
            print(f"{model_name}: {result['model_path']}")
    else:
        print(f"Training single model: YOLOv5{args.model}")
        model_path, results = trainer.train_model(
            model_size=args.model,
            pretrained=args.pretrained,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device
        )
        print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()