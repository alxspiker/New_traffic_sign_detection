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

#### Command line options

- `--data-dir`: Path to extracted data directory (default: `data_extracted`)
- `--output-dir`: Name of output directory (default: `speed_signs_extracted`)
- `--copy`: Copy files instead of moving them
- `--pattern`: Regex pattern to match in labels (default: `speed-limit`)
- `--help`: Show help message

#### What the script does

1. **Scans annotation files** in `data_extracted/annotations/` for speed-limit labels
2. **Finds matching signs** like:
   - `regulatory--maximum-speed-limit-50--g1`
   - `complementary--maximum-speed-limit-70--g1`
   - Any label containing "speed-limit"
3. **Creates output directory** with `annotations/` and `images/` subdirectories
4. **Moves/copies files** - both annotation JSON files and corresponding images
5. **Generates summary report** showing extraction statistics and sign types found

#### Output

The script creates:
- `speed_signs_extracted/annotations/` - JSON annotation files
- `speed_signs_extracted/images/` - Corresponding image files  
- `speed_signs_extracted/extraction_summary.txt` - Detailed extraction report

#### Features

- **Regex pattern matching** for flexible label searching
- **Automatic image detection** - finds corresponding images for annotations
- **Copy or move modes** - preserve originals or clean up source directory
- **Progress tracking** with progress bars
- **Detailed reporting** with extraction statistics
- **Error handling** for missing files and corrupted data

## Directory Structure After Running

```
New_traffic_sign_detection/
├── download_and_extract_dataset.py    # Dataset downloader
├── extract_speed_limit_signs.py       # Speed limit signs extractor
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── data_source/                       # Downloaded files (compressed)
│   ├── *.zip files
│   └── *.txt files
├── data_extracted/                    # Extracted MTSD dataset
│   ├── annotations/                   # All traffic sign annotations
│   ├── images/                        # All traffic sign images
│   └── *.txt files                    # MD5 checksums
└── speed_signs_extracted/             # Speed limit signs only (created by extractor)
    ├── annotations/                   # Speed limit annotations
    ├── images/                        # Speed limit images
    └── extraction_summary.txt         # Extraction report
```

## Workflow

1. **Download dataset**: Run `python download_and_extract_dataset.py`
2. **Extract speed limits**: Run `python extract_speed_limit_signs.py`
3. **Train your model**: Use files in `speed_signs_extracted/` directory

## Use Cases

- **General traffic sign detection**: Use all files in `data_extracted/`
- **Speed limit specific models**: Use files in `speed_signs_extracted/` 
- **Custom sign extraction**: Modify the pattern in `extract_speed_limit_signs.py`

## Notes

- Large files may take time to download depending on your internet connection
- The script will create the necessary directories automatically
- All zip files are extracted to the same `data_extracted` directory as specified in the dataset instructions
- Text files (MD5 checksums) are also copied to the extraction directory for reference
