# MTSD Dataset Tools

This repository contains tools for working with the Mapillary Traffic Sign Dataset (MTSD), including downloading the dataset and extracting specific types of traffic signs.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the download and extraction script:
```bash
python download_and_extract_dataset.py
```

## Scripts

### 1. Dataset Downloader (`download_and_extract_dataset.py`)

Downloads and extracts the complete MTSD dataset.

#### What the script does

1. **Creates directories:**
   - `data_source/` - stores the downloaded files
   - `data_extracted/` - stores the extracted content

2. **Downloads the following files:**
   - `mtsd_fully_annotated_md5_sums.txt`
   - `mtsd_fully_annotated_annotation.zip`
   - `mtsd_fully_annotated_images.test.zip`
   - `mtsd_fully_annotated_images.train.0.zip`
   - `mtsd_fully_annotated_images.train.1.zip`
   - `mtsd_fully_annotated_images.train.2.zip`
   - `mtsd_fully_annotated_images.val.zip`
   - `mtsd_partially_annotated_md5_sums.txt`
   - `mtsd_partially_annotated_annotation.zip`
   - `mtsd_partially_annotated_images.train.0.zip`
   - `mtsd_partially_annotated_images.train.1.zip`
   - `mtsd_partially_annotated_images.train.2.zip`
   - `mtsd_partially_annotated_images.train.3.zip`

3. **Extracts all zip files** to the `data_extracted/` directory
4. **Copies text files** (MD5 checksums) to the `data_extracted/` directory

#### Features

- **Progress bars** for downloads and extractions
- **Resume capability** - skips already downloaded files
- **Error handling** for network issues and corrupted files
- **Organized output** with clear status messages

### 2. Speed Limit Signs Extractor (`extract_speed_limit_signs.py`)

Extracts all traffic signs containing "speed-limit" in their labels from the MTSD dataset and moves them to a separate directory for training speed-limit-specific AI models.

### 3. Traffic Sign Model Trainer (`train_traffic_sign_model.py`)

**NEW!** Train custom traffic sign detection models using the MTSD dataset with modern machine learning frameworks. Creates production-ready models that can be deployed or integrated into firmware systems.

#### Usage

```bash
# Basic usage (moves files from data_extracted to speed_signs_extracted)
python extract_speed_limit_signs.py

# Copy files instead of moving them (keeps originals)
python extract_speed_limit_signs.py --copy

# Use custom directories
python extract_speed_limit_signs.py --data-dir my_data --output-dir my_speed_signs

# Use custom search pattern
python extract_speed_limit_signs.py --pattern "speed.*limit"
```

#### 3. Traffic Sign Model Trainer Usage

The training script supports various configuration options for training custom models:

```bash
# Install ML dependencies first
pip install -r requirements.txt

# Train on full MTSD dataset
python train_traffic_sign_model.py --data-dir data_extracted --epochs 100

# Train on speed limit signs only (more focused model)
python train_traffic_sign_model.py --speed-limit-only --data-dir speed_signs_extracted --epochs 150

# Custom training with specific parameters
python train_traffic_sign_model.py --data-dir data_extracted \
  --epochs 200 --batch-size 16 --image-size 640 --model-size m

# Quick test run
python train_traffic_sign_model.py --data-dir test_dataset --epochs 1
```

#### Command line options

**Speed Limit Extractor:**
- `--data-dir`: Path to extracted data directory (default: `data_extracted`)
- `--output-dir`: Name of output directory (default: `speed_signs_extracted`)
- `--copy`: Copy files instead of moving them
- `--pattern`: Regex pattern to match in labels (default: `speed-limit`)
- `--help`: Show help message

**Model Trainer:**
- `--data-dir`: Path to dataset directory (required)
- `--speed-limit-only`: Train only on speed limit signs
- `--output-dir`: Output directory for training results (default: `training_output`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 16)
- `--image-size`: Input image size (default: 640)
- `--model-size`: YOLO model size: n, s, m, l, x (default: n)
- `--learning-rate`: Learning rate (default: 0.01)
- `--val-split`: Validation split ratio (default: 0.2)
- `--force-cpu`: Force CPU training even if GPU is available

#### What the script does

1. **Scans annotation files** in `data_extracted/annotations/` for speed-limit labels
2. **Finds matching signs** like:
   - `regulatory--maximum-speed-limit-50--g1`
   - `complementary--maximum-speed-limit-70--g1`
   - Any label containing "speed-limit"
3. **Creates output directory** with `annotations/` and `images/` subdirectories
4. **Moves/copies files** - both annotation JSON files and corresponding images
5. **Generates summary report** showing extraction statistics and sign types found

#### What the training script does

1. **Loads MTSD dataset** or extracted speed limit signs data
2. **Analyzes annotations** to create class mappings and dataset statistics  
3. **Prepares data** in YOLO format with proper train/validation splits
4. **Trains YOLOv8 model** with configurable parameters and GPU acceleration
5. **Evaluates performance** with validation metrics (mAP, precision, recall)
6. **Exports models** in multiple formats:
   - PyTorch (.pt) - for Python deployment
   - ONNX (.onnx) - for cross-platform inference
   - TensorFlow Lite (.tflite) - for mobile/edge devices
   - Binary (.bin) - custom format for firmware integration
7. **Creates comprehensive report** with training details and usage instructions

#### Output

**Speed Limit Extractor creates:**
- `speed_signs_extracted/annotations/` - JSON annotation files
- `speed_signs_extracted/images/` - Corresponding image files  
- `speed_signs_extracted/extraction_summary.txt` - Detailed extraction report

**Model Trainer creates:**
- `training_output/models/` - Trained model files
- `training_output/exports/` - Exported models in multiple formats
- `training_output/yolo_dataset/` - Processed YOLO format dataset
- `training_output/yolo_training/` - Training logs and intermediate results
- `training_output/plots/` - Training visualization plots
- `training_output/training_report.txt` - Comprehensive training report

#### Features

**Speed Limit Extractor:**
- **Regex pattern matching** for flexible label searching
- **Automatic image detection** - finds corresponding images for annotations
- **Copy or move modes** - preserve originals or clean up source directory
- **Progress tracking** with progress bars
- **Detailed reporting** with extraction statistics
- **Error handling** for missing files and corrupted data

**Model Trainer:**
- **Modern YOLOv8 architecture** - state-of-the-art object detection
- **GPU acceleration** - automatic CUDA detection and usage
- **Flexible dataset support** - works with full MTSD or extracted subsets
- **Comprehensive evaluation** - mAP, precision, recall metrics
- **Multiple export formats** - PyTorch, ONNX, TensorFlow Lite, custom binary
- **Configurable training** - epochs, batch size, learning rate, model size
- **Binary model export** - custom format designed for firmware integration
- **Production ready** - includes model validation and deployment guidance

## Directory Structure After Running

```
New_traffic_sign_detection/
├── download_and_extract_dataset.py    # Dataset downloader
├── extract_speed_limit_signs.py       # Speed limit signs extractor
├── train_traffic_sign_model.py        # NEW! AI model trainer
├── test_training_script.py            # Test script for trainer validation
├── requirements.txt                    # Python dependencies (updated with ML libs)
├── README.md                          # This file
├── data_source/                       # Downloaded files (compressed)
│   ├── *.zip files
│   └── *.txt files
├── data_extracted/                    # Extracted MTSD dataset
│   ├── annotations/                   # All traffic sign annotations
│   ├── images/                        # All traffic sign images
│   └── *.txt files                    # MD5 checksums
├── speed_signs_extracted/             # Speed limit signs only (created by extractor)
│   ├── annotations/                   # Speed limit annotations
│   ├── images/                        # Speed limit images
│   └── extraction_summary.txt         # Extraction report
├── training_output/                   # Model training results (created by trainer)
│   ├── models/                        # Trained PyTorch models
│   ├── exports/                       # Exported models (ONNX, TFLite, binary)
│   ├── yolo_dataset/                  # Processed YOLO format dataset
│   ├── yolo_training/                 # Training logs and weights
│   ├── plots/                         # Training visualization plots
│   └── training_report.txt            # Training summary report
└── test_dataset/                      # Dummy dataset for testing
    ├── images/                        # Test images
    └── mtsd_v2_fully_annotated/       # Test annotations
        └── annotations/
```

## Workflow

### Quick Start (Recommended)
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Download dataset**: `python download_and_extract_dataset.py`
3. **Train model**: `python train_traffic_sign_model.py --data-dir data_extracted --epochs 100`

### Advanced Workflow
1. **Download dataset**: Run `python download_and_extract_dataset.py`
2. **Extract speed limits** (optional): Run `python extract_speed_limit_signs.py`
3. **Train your model**: 
   - Full dataset: `python train_traffic_sign_model.py --data-dir data_extracted`
   - Speed limits only: `python train_traffic_sign_model.py --speed-limit-only --data-dir speed_signs_extracted`
4. **Deploy model**: Use exported models from `training_output/exports/`

## Use Cases

- **General traffic sign detection**: Use full dataset with `train_traffic_sign_model.py --data-dir data_extracted`
- **Speed limit specific models**: Use extracted data with `train_traffic_sign_model.py --speed-limit-only --data-dir speed_signs_extracted` 
- **Custom sign extraction**: Modify the pattern in `extract_speed_limit_signs.py`
- **Firmware integration**: Use the binary model export (.bin) for embedded systems
- **Mobile deployment**: Use TensorFlow Lite export (.tflite) for mobile apps
- **Production inference**: Use ONNX export (.onnx) for cross-platform deployment

## Model Integration & Deployment

The training script creates models in multiple formats for different deployment scenarios:

### Binary Model Format (.bin)
- **Purpose**: Designed for firmware integration (similar to the reference HIK Vision toolkit)
- **Structure**: Custom binary format with embedded metadata
- **Contains**: Model weights, class names, configuration parameters
- **Usage**: Can be integrated into firmware systems for embedded inference

### Standard Formats
- **PyTorch (.pt)**: For Python-based inference and further training
- **ONNX (.onnx)**: For cross-platform deployment and optimization
- **TensorFlow Lite (.tflite)**: For mobile and edge device deployment

### Performance Characteristics
- **YOLOv8n (nano)**: Fast inference, lower accuracy (~30 FPS on CPU)
- **YOLOv8s (small)**: Balanced speed/accuracy (~20 FPS on CPU)  
- **YOLOv8m (medium)**: Better accuracy, slower inference (~10 FPS on CPU)
- **YOLOv8l/x (large/xlarge)**: Best accuracy, requires GPU for real-time inference

## Notes

- Large files may take time to download depending on your internet connection
- The script will create the necessary directories automatically
- All zip files are extracted to the same `data_extracted` directory as specified in the dataset instructions
- Text files (MD5 checksums) are also copied to the extraction directory for reference
