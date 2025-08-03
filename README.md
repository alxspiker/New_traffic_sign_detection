# MTSD Dataset Downloader

This script automatically downloads and extracts the Mapillary Traffic Sign Dataset (MTSD) files.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the download and extraction script:
```bash
python download_and_extract_dataset.py
```

## What the script does

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

## Features

- **Progress bars** for downloads and extractions
- **Resume capability** - skips already downloaded files
- **Error handling** for network issues and corrupted files
- **Organized output** with clear status messages

## Directory Structure After Running

```
New_traffic_sign_detection/
├── download_and_extract_dataset.py
├── requirements.txt
├── README.md
├── data_source/              # Downloaded files (compressed)
│   ├── *.zip files
│   └── *.txt files
└── data_extracted/           # Extracted content
    ├── annotations/
    ├── images/
    └── *.txt files
```

## Notes

- Large files may take time to download depending on your internet connection
- The script will create the necessary directories automatically
- All zip files are extracted to the same `data_extracted` directory as specified in the dataset instructions
- Text files (MD5 checksums) are also copied to the extraction directory for reference
