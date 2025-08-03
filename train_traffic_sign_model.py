#!/usr/bin/env python3
"""
Traffic Sign Detection Model Training Script
Complete pipeline for training custom traffic sign detection models using MTSD dataset

Usage:
    python3 train_traffic_sign_model.py --data-dir data_extracted --epochs 100
    python3 train_traffic_sign_model.py --speed-limit-only --data-dir speed_signs_extracted
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import shutil

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms
    import numpy as np
    from PIL import Image, ImageDraw
    import cv2
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import ultralytics
    from ultralytics import YOLO
except ImportError as e:
    print(f"❌ Missing required ML libraries. Please install them:")
    print("pip install torch torchvision ultralytics opencv-python pillow scikit-learn matplotlib numpy tqdm")
    sys.exit(1)


class TrafficSignTrainer:
    """Main class for training traffic sign detection models"""
    
    def __init__(self, config: Dict):
        """Initialize the trainer with configuration"""
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.setup_device()
        
        # Model and training state
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.class_names = []
        self.class_to_idx = {}
        
        print("🚗 Traffic Sign Detection Model Trainer")
        print("=" * 50)
        print(f"📁 Data directory: {self.config['data_dir']}")
        print(f"🎯 Output directory: {self.config['output_dir']}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Model type: {self.config['model_type']}")
        print()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['output_dir']) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"training_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Setup output directories"""
        self.output_dir = Path(self.config['output_dir'])
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"
        self.plots_dir = self.output_dir / "plots"
        self.exports_dir = self.output_dir / "exports"
        
        # Create directories
        for dir_path in [self.models_dir, self.logs_dir, self.plots_dir, self.exports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_device(self):
        """Setup computing device (GPU/CPU)"""
        if torch.cuda.is_available() and not self.config.get('force_cpu', False):
            self.device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU")
    
    def load_annotations(self) -> List[Dict]:
        """Load and parse annotation files"""
        data_dir = Path(self.config['data_dir'])
        
        # Handle different directory structures
        if self.config.get('speed_limit_only', False):
            # Speed limit extracted data
            annotations_dir = data_dir / "annotations"
            images_dir = data_dir / "images"
        else:
            # Full MTSD dataset
            annotations_dir = data_dir / "mtsd_v2_fully_annotated" / "annotations"
            images_dir = data_dir / "images"
        
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        print(f"📋 Loading annotations from: {annotations_dir}")
        print(f"🖼️  Loading images from: {images_dir}")
        
        annotations = []
        json_files = list(annotations_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No annotation files found in {annotations_dir}")
        
        print(f"🔍 Processing {len(json_files)} annotation files...")
        
        for json_file in tqdm(json_files, desc="Loading annotations"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Find corresponding image
                base_name = json_file.stem
                image_path = None
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_path = images_dir / f"{base_name}{ext}"
                    if potential_path.exists():
                        image_path = potential_path
                        break
                
                if not image_path or not image_path.exists():
                    self.logger.warning(f"No image found for {json_file.name}")
                    continue
                
                # Parse annotation data
                if 'objects' in data and data['objects']:
                    annotation_entry = {
                        'image_path': str(image_path),
                        'annotation_path': str(json_file),
                        'width': data.get('width', 0),
                        'height': data.get('height', 0),
                        'objects': data['objects']
                    }
                    annotations.append(annotation_entry)
                
            except Exception as e:
                self.logger.warning(f"Failed to process {json_file.name}: {e}")
                continue
        
        print(f"✅ Loaded {len(annotations)} valid annotations")
        return annotations
    
    def create_class_mapping(self, annotations: List[Dict]) -> Tuple[List[str], Dict[str, int]]:
        """Create class name mapping from annotations"""
        all_labels = set()
        
        for annotation in annotations:
            for obj in annotation['objects']:
                if 'label' in obj:
                    all_labels.add(obj['label'])
        
        # Sort for consistent ordering
        class_names = sorted(list(all_labels))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"🏷️  Found {len(class_names)} unique traffic sign classes")
        
        # Show some examples
        print("📋 Example classes:")
        for i, class_name in enumerate(class_names[:10]):
            print(f"   {i}: {class_name}")
        if len(class_names) > 10:
            print(f"   ... and {len(class_names) - 10} more")
        
        return class_names, class_to_idx
    
    def prepare_yolo_format(self, annotations: List[Dict]) -> str:
        """Prepare data in YOLO format and return dataset config path"""
        yolo_dir = self.output_dir / "yolo_dataset"
        
        # Create YOLO directory structure
        for split in ['train', 'val']:
            (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Split data
        train_annotations, val_annotations = train_test_split(
            annotations, 
            test_size=self.config['val_split'], 
            random_state=42
        )
        
        print(f"📊 Dataset split:")
        print(f"   Training: {len(train_annotations)} images")
        print(f"   Validation: {len(val_annotations)} images")
        
        # Process each split
        for split_name, split_annotations in [('train', train_annotations), ('val', val_annotations)]:
            print(f"🔧 Preparing {split_name} data...")
            
            for annotation in tqdm(split_annotations, desc=f"Processing {split_name}"):
                # Copy image
                image_path = Path(annotation['image_path'])
                dest_image = yolo_dir / split_name / 'images' / image_path.name
                shutil.copy2(image_path, dest_image)
                
                # Create YOLO label file
                label_file = yolo_dir / split_name / 'labels' / f"{image_path.stem}.txt"
                
                with open(label_file, 'w') as f:
                    for obj in annotation['objects']:
                        if 'bbox' in obj and 'label' in obj:
                            bbox = obj['bbox']
                            class_idx = self.class_to_idx[obj['label']]
                            
                            # Convert to YOLO format (normalized xywh)
                            x_center = (bbox['xmin'] + bbox['xmax']) / 2 / annotation['width']
                            y_center = (bbox['ymin'] + bbox['ymax']) / 2 / annotation['height']
                            width = (bbox['xmax'] - bbox['xmin']) / annotation['width']
                            height = (bbox['ymax'] - bbox['ymin']) / annotation['height']
                            
                            f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Create dataset config file
        config_file = yolo_dir / "dataset.yaml"
        with open(config_file, 'w') as f:
            f.write(f"path: {yolo_dir.absolute()}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write(f"nc: {len(self.class_names)}\n")
            f.write(f"names: {self.class_names}\n")
        
        print(f"✅ YOLO dataset prepared at: {yolo_dir}")
        return str(config_file)
    
    def train_yolo_model(self, dataset_config: str):
        """Train YOLO model"""
        print(f"\n🚀 Starting YOLO model training...")
        
        # Initialize model
        model_size = self.config.get('model_size', 'n')  # n, s, m, l, x
        model = YOLO(f'yolov8{model_size}.pt')  # Load pretrained model
        
        # Training parameters
        train_params = {
            'data': dataset_config,
            'epochs': self.config['epochs'],
            'imgsz': self.config['image_size'],
            'batch': self.config['batch_size'],
            'device': 'cpu' if self.device.type == 'cpu' else 0,
            'project': str(self.output_dir),
            'name': 'yolo_training',
            'save': True,
            'save_period': 10,
            'cache': False,
            'workers': self.config.get('workers', 8),
            'patience': self.config.get('patience', 50),
            'lr0': self.config.get('learning_rate', 0.01),
            'weight_decay': self.config.get('weight_decay', 0.0005),
        }
        
        print(f"🎯 Training parameters:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
        
        # Train the model
        results = model.train(**train_params)
        
        # Save trained model
        best_model_path = self.output_dir / "yolo_training" / "weights" / "best.pt"
        final_model_path = self.models_dir / "traffic_signs_best.pt"
        
        if best_model_path.exists():
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ Best model saved to: {final_model_path}")
        
        self.model = model
        return results
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("❌ No model available for evaluation")
            return
        
        print(f"\n📊 Evaluating model performance...")
        
        # Run validation
        metrics = self.model.val()
        
        # Print key metrics
        if hasattr(metrics, 'box'):
            print(f"📈 Key Metrics:")
            print(f"   mAP50: {metrics.box.map50:.4f}")
            print(f"   mAP50-95: {metrics.box.map:.4f}")
            print(f"   Precision: {metrics.box.mp:.4f}")
            print(f"   Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_models(self):
        """Export trained model in multiple formats"""
        if self.model is None:
            print("❌ No model available for export")
            return
        
        print(f"\n📦 Exporting model in multiple formats...")
        
        export_formats = [
            ('pt', 'PyTorch'),
            ('onnx', 'ONNX'),
            ('torchscript', 'TorchScript'),
            ('tflite', 'TensorFlow Lite'),
        ]
        
        exported_files = {}
        
        for format_ext, format_name in export_formats:
            try:
                print(f"🔧 Exporting to {format_name}...")
                export_path = self.model.export(format=format_ext, imgsz=self.config['image_size'])
                
                # Move to exports directory
                if isinstance(export_path, str):
                    export_path = Path(export_path)
                
                dest_path = self.exports_dir / export_path.name
                if export_path.exists():
                    shutil.copy2(export_path, dest_path)
                    exported_files[format_ext] = dest_path
                    print(f"✅ {format_name} exported to: {dest_path}")
                
            except Exception as e:
                print(f"⚠️  Failed to export to {format_name}: {e}")
        
        # Create binary model file (for firmware integration like the reference script)
        if 'pt' in exported_files:
            try:
                print(f"🔧 Creating binary model file...")
                binary_path = self.exports_dir / "traffic_signs.bin"
                
                # Load the PyTorch model and save as binary
                model_data = torch.load(exported_files['pt'], map_location='cpu')
                
                # Extract just the model weights as binary data
                model_bytes = bytearray()
                
                # Add model metadata header
                header = f"TRAFFIC_SIGN_MODEL_V1".encode('utf-8')
                model_bytes.extend(header)
                model_bytes.extend(b'\x00' * (64 - len(header)))  # Pad to 64 bytes
                
                # Add model info
                info = {
                    'num_classes': len(self.class_names),
                    'image_size': self.config['image_size'],
                    'model_type': 'yolov8',
                    'classes': self.class_names
                }
                info_json = json.dumps(info).encode('utf-8')
                info_size = len(info_json)
                model_bytes.extend(info_size.to_bytes(4, 'little'))
                model_bytes.extend(info_json)
                
                # Save the complete model as pickle bytes for easy loading
                import pickle
                model_pickle = pickle.dumps(model_data)
                model_bytes.extend(len(model_pickle).to_bytes(4, 'little'))
                model_bytes.extend(model_pickle)
                
                with open(binary_path, 'wb') as f:
                    f.write(model_bytes)
                
                size_mb = len(model_bytes) / (1024 * 1024)
                print(f"✅ Binary model created: {binary_path} ({size_mb:.1f} MB)")
                exported_files['bin'] = binary_path
                
            except Exception as e:
                print(f"⚠️  Failed to create binary model: {e}")
        
        return exported_files
    
    def create_training_report(self, training_results, exported_files: Dict):
        """Create comprehensive training report"""
        report_file = self.output_dir / "training_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("TRAFFIC SIGN DETECTION MODEL TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration
            f.write("TRAINING CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Dataset info
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total classes: {len(self.class_names)}\n")
            f.write(f"Training images: {int(len(self.train_dataset) if self.train_dataset else 0)}\n")
            f.write(f"Validation images: {int(len(self.val_dataset) if self.val_dataset else 0)}\n")
            f.write("\n")
            
            # Class names
            f.write("DETECTED CLASSES:\n")
            f.write("-" * 30 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{i:3d}: {class_name}\n")
            f.write("\n")
            
            # Exported files
            f.write("EXPORTED MODEL FILES:\n")
            f.write("-" * 30 + "\n")
            for format_type, file_path in exported_files.items():
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    f.write(f"{format_type.upper()}: {file_path} ({size_mb:.1f} MB)\n")
            f.write("\n")
            
            # Usage instructions
            f.write("USAGE INSTRUCTIONS:\n")
            f.write("-" * 30 + "\n")
            f.write("1. PyTorch model: Load with torch.load('traffic_signs_best.pt')\n")
            f.write("2. ONNX model: Use with ONNX runtime or convert to other formats\n")
            f.write("3. Binary model: Custom format for firmware integration\n")
            f.write("4. TensorFlow Lite: Use for mobile/edge deployment\n")
            f.write("\n")
            
            f.write("INTEGRATION NOTES:\n")
            f.write("-" * 30 + "\n")
            f.write("- The binary model format is designed for firmware integration\n")
            f.write("- Model expects RGB images at {}x{} resolution\n".format(
                self.config['image_size'], self.config['image_size']))
            f.write("- Output format: class_id, confidence, x, y, width, height\n")
            f.write("- Use the class names list for label mapping\n")
        
        print(f"📄 Training report saved to: {report_file}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            # Load and prepare data
            print("🔍 Step 1: Loading annotations...")
            annotations = self.load_annotations()
            
            print("🏷️  Step 2: Creating class mapping...")
            self.class_names, self.class_to_idx = self.create_class_mapping(annotations)
            
            print("📋 Step 3: Preparing YOLO format dataset...")
            dataset_config = self.prepare_yolo_format(annotations)
            
            print("🚀 Step 4: Training model...")
            training_results = self.train_yolo_model(dataset_config)
            
            print("📊 Step 5: Evaluating model...")
            metrics = self.evaluate_model()
            
            print("📦 Step 6: Exporting models...")
            exported_files = self.export_models()
            
            print("📄 Step 7: Creating training report...")
            self.create_training_report(training_results, exported_files)
            
            print(f"\n🎉 Training pipeline completed successfully!")
            print(f"📁 All outputs saved to: {self.output_dir}")
            print(f"🎯 Best model: {self.models_dir / 'traffic_signs_best.pt'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            print(f"❌ Training failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train traffic sign detection models using MTSD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on full MTSD dataset
  python3 train_traffic_sign_model.py --data-dir data_extracted --epochs 100
  
  # Train on speed limit signs only
  python3 train_traffic_sign_model.py --speed-limit-only --data-dir speed_signs_extracted
  
  # Custom training with specific parameters
  python3 train_traffic_sign_model.py --epochs 200 --batch-size 16 --image-size 640
        """
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--speed-limit-only', action='store_true',
                      help='Train only on speed limit signs (use with speed_signs_extracted)')
    parser.add_argument('--output-dir', type=str, default='training_output',
                      help='Output directory for training results (default: training_output)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                      help='Batch size for training (default: 16)')
    parser.add_argument('--image-size', type=int, default=640,
                      help='Input image size (default: 640)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='Learning rate (default: 0.01)')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='Validation split ratio (default: 0.2)')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='yolo',
                      choices=['yolo'], help='Model architecture (default: yolo)')
    parser.add_argument('--model-size', type=str, default='n',
                      choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLO model size: n(ano), s(mall), m(edium), l(arge), x(large) (default: n)')
    
    # System arguments
    parser.add_argument('--force-cpu', action='store_true',
                      help='Force CPU training even if GPU is available')
    parser.add_argument('--workers', type=int, default=8,
                      help='Number of data loading workers (default: 8)')
    parser.add_argument('--patience', type=int, default=50,
                      help='Early stopping patience (default: 50)')
    
    args = parser.parse_args()
    
    # Validate arguments
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        if args.speed_limit_only:
            print("💡 Tip: Run extract_speed_limit_signs.py first to create speed_signs_extracted")
        else:
            print("💡 Tip: Run download_and_extract_dataset.py first to download the MTSD dataset")
        sys.exit(1)
    
    # Create configuration
    config = {
        'data_dir': str(data_dir),
        'speed_limit_only': args.speed_limit_only,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'val_split': args.val_split,
        'model_type': args.model_type,
        'model_size': args.model_size,
        'force_cpu': args.force_cpu,
        'workers': args.workers,
        'patience': args.patience,
        'weight_decay': 0.0005,
    }
    
    # Initialize and run trainer
    trainer = TrafficSignTrainer(config)
    
    print(f"🚀 Starting training with configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    success = trainer.run_training_pipeline()
    
    if success:
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Check {args.output_dir}/ for all outputs")
        sys.exit(0)
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)