#!/usr/bin/env python3
"""
Automated script to download and extract MTSD (Mapillary Traffic Sign Dataset) files.
This script downloads all the dataset files and extracts them directly into the data_source directory.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import hashlib
from tqdm import tqdm

# Dataset URLs and corresponding filenames - FULLY ANNOTATED ONLY
DATASET_FILES = [
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An8RSqTVAOhQUZHE5NgIL-tgliMqFe4h3ieVXSDNI-ilkh922n_TtB2VpiW-SzAHEYLiqk3Y3FU8pM-DskkOMCvFGh9RJAxxqi3RI2JXDvbQfq4xYwVstPYKlT44GhxzbDbdW_OG0lyY.txt?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=AdnwNEWtd-ufOGxJEhED0pgxr1jKvAwlIZFJbSDRV06SO1dSMoENAvo196ayTY6zbdvdCI0rMHSOSgAyLdhvl0an&ccb=10-5&oh=00_AfSLK3exCT_kIY7TetgaCAOVmLu7jEM76L_ZPZNDPTLHsw&oe=68B62345&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_md5_sums.txt"
    },
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An90x89nHvauCK1fqMJ8110KeTjNo5Si7rzhvwIMCu5xI9_GhWBGOIXaFvu6o53NuNpBMzdC9qsjAVR8sLv8m6WoFfn6Qd4NjMYKNW4NCKVp6gx3MhZtwf3cZR94wFhou5lPI0hGUw.zip?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=AdmNvL-9zE1ZE6kjMMg-rlVrDofyMfx9-k60rNQOMiWo0jRINqRwg-fuxE7Lm4i1VJQ92iLKgaQDDZOlMqyUjI4Q&ccb=10-5&oh=00_AfRAU0LMal3AmUHg9wtT6QLpiJoIAW03V6BdXyfOwgDCVQ&oe=68B61488&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_annotation.zip"
    },
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An9eB8zXYW473CiQYW9CPvGx1Ho-fkQvkini3ddExpFOz47aWs4ydBSvK-ZhOPu7ikASQmZvX0zyXhmJzBr6CDZE5ZkUhvJ44h7mV2NT4cSRbR837J9mHJosreQRJJdGaVDR26EAjLPL.zip?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=AdkbzOcRZUbq5GhWyKteqlvKpmO3TE0zo9ChD1cvPJFbFa6nMrZijEV6QiVu-0geGPFu6Vx9hetV8MLOuKEOkAiG&ccb=10-5&oh=00_AfQwavHpBEmX8MLYiFt2SshL3aMcJIApBRSNLeL8_oBgRg&oe=68B61E1C&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_images.test.zip"
    },
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An_WKGcw-ICowA_xAgTEU_E-pAYdybyzT-9Pwi8JtelanWnNRKONV1DTAZPEAsGNDWlFpYDi16km1stDN47ip-quE77cfkv3aERdMIRahysGgspb6DlCgrabPSTFJI3tZ9EMRatRC6ZmjQytVcY.zip?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=AdmWBTyhEaaXs7ZpBqWgxvCcLosLagzbwsjhhH542r6y482uprLVXbMasDW1qzqSSkR43eFGFr-5ZR5WqUZnFf1e&ccb=10-5&oh=00_AfTijduIsx5VI3AQC-wjgtrMr7QQuxalruq34_dQJbVlYA&oe=68B61A72&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_images.train.0.zip"
    },
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An8VtaI-ldaSOc5HcLFVQ6SPvHDt50hLG1kga0nUfswldLu1J9dsOx6ynZicRUuXR_TvsczpplOqQEa7ppT4JwUzI0ZNQCHmhtkfT5tjdNJY55Ud6eXplvq59PjOx55d2EbIxYpO9vhR-BcflQ.zip?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=AdlCrZ6brZhq1N4oJyJGKCBhUYOUmlAlLxkPOCCL73q_21KF56XyupvwfgbTO7O2Ohht3l-AA7bstogDmez_tAhV&ccb=10-5&oh=00_AfRNmpspYq77fVkYq5EWNlumlPvgLpxpX6dkDnKjrIcqwA&oe=68B5F41E&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_images.train.1.zip"
    },
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An9BwhO8zTAy6jGrwy71BjIBtMK8K5RkpIJguP7DVnpJK2TfKDlfxXj8mCxRJss4zzfaaKi2idqbQOtYJ740TPCI7w7hL8V7goknzuO0ZFPLywDCKIB7i64lCiSUNYXLqeS8mC7EkiU5hYAfOcI.zip?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=Adnp5ge6-eWeKBRiAsMrlRfPLo7XfdepjOR-J4OvL7PKE0BOwWX5Tj1b2Lug2h_6znFbCqX8kl3PdaDyFB0WUswq&ccb=10-5&oh=00_AfTUwKQ8MnaqckXH4f4oj2bLvghxO3f2Mbtjj2rq5Fm82w&oe=68B615DB&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_images.train.2.zip"
    },
    {
        "url": "https://scontent.fyvr3-1.fna.fbcdn.net/m1/v/t6/An9tl3SwDwRFQ2z9tAag26brYamdQZmHnVoxfNTz_Iass-zZLWM-HryqW44UeqbLWd-EkXVIP-ZQQfg3F7dmQYlnu1wjzCARviaJMBHgtLH4gTAeW6msFbEXA3_NIZBtdP7Gg8dt5Ewl.zip?_nc_gid=OIShwu9bjMMdACwunpCRIg&_nc_oc=AdnlE1AklQs87IVryPyJNejd4H8Q6BfERGUPYUx5cWpNVGZL3JrxZdRcjve28nxb44fGz24ftzzPc0rbP84975sW&ccb=10-5&oh=00_AfSIOFmLKYdJeYMYBdq9Ea4XrrzQRGrALHKsQ1goeTz30g&oe=68B5FD27&_nc_sid=6de079",
        "filename": "mtsd_fully_annotated_images.val.zip"
    }
]

def create_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent
    data_source_dir = base_dir / "data_source"
    data_extracted_dir = base_dir / "data_extracted"
    
    data_source_dir.mkdir(exist_ok=True)
    data_extracted_dir.mkdir(exist_ok=True)
    
    return data_source_dir, data_extracted_dir

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def verify_file_integrity(file_path, expected_md5=None):
    """Verify file integrity using MD5 hash."""
    if not file_path.exists():
        return False
    
    if expected_md5:
        actual_md5 = calculate_md5(file_path)
        if actual_md5.lower() == expected_md5.lower():
            print(f"✅ {file_path.name} - Checksum verified")
            return True
        else:
            print(f"❌ {file_path.name} - Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
            return False
    return True

def parse_md5_file(md5_file_path):
    """Parse MD5 checksum file and return dictionary of filename -> hash."""
    checksums = {}
    if md5_file_path.exists():
        try:
            with open(md5_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) >= 2:
                            # MD5 format can be "hash filename" or "hash *filename"
                            md5_hash = parts[0]
                            filename = parts[1].lstrip('*')
                            
                            # Handle the filename mismatch - checksum files use mtsd_v2_ prefix
                            # but our downloaded files don't have this prefix
                            if filename.startswith('mtsd_v2_'):
                                # Remove the mtsd_v2_ prefix to match our actual filenames
                                actual_filename = filename.replace('mtsd_v2_', 'mtsd_')
                                checksums[actual_filename] = md5_hash
                            
                            # Also store with original name in case it matches
                            checksums[filename] = md5_hash
                            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse MD5 file {md5_file_path.name}: {e}")
    return checksums

def download_file(url, filename, download_dir, expected_md5=None, force_redownload=False):
    """Download a file with progress bar and optional checksum verification."""
    file_path = download_dir / filename
    
    # Check if file exists and is valid
    if file_path.exists() and not force_redownload:
        if expected_md5:
            if verify_file_integrity(file_path, expected_md5):
                print(f"✓ {filename} already exists and is verified, skipping download")
                return file_path
            else:
                print(f"⚠️  {filename} exists but checksum failed, re-downloading...")
        else:
            print(f"✓ {filename} already exists, skipping download")
            return file_path
    
    print(f"📥 Downloading {filename}...")
    
    try:
        # Get file size for progress bar
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        
        # Verify downloaded file if checksum provided
        if expected_md5:
            if verify_file_integrity(file_path, expected_md5):
                print(f"✅ Successfully downloaded and verified {filename}")
            else:
                print(f"❌ Downloaded {filename} but checksum verification failed!")
                return None
        else:
            print(f"✅ Successfully downloaded {filename}")
        
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None

def extract_zip_file(zip_path, extract_dir):
    """Extract a zip file to the specified directory."""
    if not zip_path.exists():
        print(f"❌ File {zip_path.name} not found, skipping extraction")
        return False
    
    print(f"📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()
            
            with tqdm(total=len(file_list), desc=f"Extracting {zip_path.name}") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_dir)
                    pbar.update(1)
        
        print(f"✅ Successfully extracted {zip_path.name}")
        return True
        
    except zipfile.BadZipFile:
        print(f"❌ Error: {zip_path.name} is not a valid zip file")
        return False
    except Exception as e:
        print(f"❌ Error extracting {zip_path.name}: {e}")
        return False

def main():
    """Main function to download and extract all dataset files."""
    print("🚦 MTSD Dataset Downloader and Extractor")
    print("=" * 50)
    
    # Create directories
    data_source_dir, data_extracted_dir = create_directories()
    print(f"📁 Data source directory: {data_source_dir}")
    print(f"📁 Data extraction directory: {data_extracted_dir}")
    print()
    
    # Download all files
    print("🌐 Starting downloads...")
    downloaded_files = []
    
    # First, download MD5 checksum files
    md5_files = [f for f in DATASET_FILES if f['filename'].endswith('_md5_sums.txt')]
    checksums = {}
    
    print("📋 Downloading checksum files first...")
    for file_info in md5_files:
        print(f"Downloading {file_info['filename']}")
        downloaded_file = download_file(
            file_info['url'], 
            file_info['filename'], 
            data_source_dir
        )
        if downloaded_file:
            downloaded_files.append(downloaded_file)
            # Parse the MD5 file to get checksums
            file_checksums = parse_md5_file(downloaded_file)
            checksums.update(file_checksums)
    
    print(f"📋 Loaded {len(checksums)} checksums for verification")
    print()
    
    # Download other files with checksum verification
    non_md5_files = [f for f in DATASET_FILES if not f['filename'].endswith('_md5_sums.txt')]
    verified_during_download = set()  # Track files already verified during download
    
    for i, file_info in enumerate(non_md5_files, 1):
        filename = file_info['filename']
        expected_md5 = checksums.get(filename)
        
        print(f"\n[{i}/{len(non_md5_files)}] Processing {filename}")
        if expected_md5:
            print(f"🔍 Expected MD5: {expected_md5}")
        else:
            print("⚠️  No checksum available for verification")
        
        downloaded_file = download_file(
            file_info['url'], 
            filename, 
            data_source_dir,
            expected_md5
        )
        
        if downloaded_file:
            downloaded_files.append(downloaded_file)
            # If file was verified during download (either existed and was verified, or was downloaded and verified)
            if expected_md5:
                verified_during_download.add(filename)
        elif expected_md5:  # If download failed and we have checksum, this is critical
            print(f"❌ Critical: Failed to download {filename} with valid checksum!")
    
    print(f"\n✅ Download phase completed. {len(downloaded_files)} files ready for extraction.")
    
    # Final integrity check (only for files not already verified)
    print("\n🔍 Final integrity check...")
    verified_files = []
    for file_path in downloaded_files:
        if file_path.suffix.lower() != '.txt':  # Skip MD5 files
            if file_path.name in verified_during_download:
                print(f"✅ {file_path.name} - Already verified, skipping")
                verified_files.append(file_path)
            else:
                expected_md5 = checksums.get(file_path.name)
                if expected_md5:
                    if verify_file_integrity(file_path, expected_md5):
                        verified_files.append(file_path)
                    else:
                        print(f"⚠️  File {file_path.name} failed final integrity check!")
                else:
                    print(f"⚠️  No checksum available for {file_path.name}, proceeding anyway")
                    verified_files.append(file_path)
    
    # Extract zip files into data_extracted directory
    print("\n📦 Extracting zip files into data_extracted directory...")
    zip_files = [f for f in verified_files if f.suffix.lower() == '.zip']
    
    if zip_files:
        print(f"Found {len(zip_files)} verified zip files to extract.")
        
        for i, zip_file in enumerate(zip_files, 1):
            print(f"\n[{i}/{len(zip_files)}] Extracting {zip_file.name}")
            extract_zip_file(zip_file, data_extracted_dir)  # Extract into data_extracted
    else:
        print("No verified zip files found to extract.")
    
    # Copy text files to extracted directory
    txt_files = [f for f in downloaded_files if f.suffix.lower() == '.txt']
    if txt_files:
        print(f"\n📄 Copying {len(txt_files)} text files to extraction directory...")
        for txt_file in txt_files:
            dest_path = data_extracted_dir / txt_file.name
            import shutil
            shutil.copy2(txt_file, dest_path)
            print(f"✅ Copied {txt_file.name}")
    
    # Final summary
    failed_files = len([f for f in DATASET_FILES if not f['filename'].endswith('_md5_sums.txt')]) - len(verified_files)
    
    print("\n🎉 All operations completed!")
    print(f"📊 Summary:")
    print(f"   • Total files: {len(DATASET_FILES)}")
    print(f"   • Successfully downloaded: {len(downloaded_files)}")
    print(f"   • Integrity verified: {len(verified_files) + 1}")  # +1 for MD5 file
    print(f"   • Failed/Corrupted: {failed_files}")
    print(f"   • Extracted zip files: {len(zip_files)}")
    print(f"   • Source directory: {data_source_dir}")
    print(f"   • Extracted directory: {data_extracted_dir}")
    
    if failed_files > 0:
        print(f"\n⚠️  Warning: {failed_files} files failed to download or verify correctly!")
        print("   Consider running the script again to retry failed downloads.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
