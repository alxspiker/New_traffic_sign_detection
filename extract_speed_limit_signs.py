#!/usr/bin/env python3
"""
Script to extract speed-limit labeled traffic signs from the MTSD dataset.
Moves annotations and corresponding images containing speed-limit signs to a separate directory.
"""

import os
import json
import re
import shutil
from pathlib import Path
from typing import Set, List, Dict, Any
import argparse
from tqdm import tqdm


def find_speed_limit_annotations(annotations_dir: Path, pattern: str = r"speed-limit") -> Dict[str, List[str]]:
    """
    Find all annotation files containing objects with speed-limit labels.
    
    Args:
        annotations_dir: Path to the annotations directory
        pattern: Regex pattern to match in labels (default: "speed-limit")
    
    Returns:
        Dictionary mapping annotation filenames to lists of matching labels
    """
    speed_limit_files = {}
    pattern_regex = re.compile(pattern, re.IGNORECASE)
    
    if not annotations_dir.exists():
        print(f"❌ Annotations directory not found: {annotations_dir}")
        return speed_limit_files
    
    json_files = list(annotations_dir.glob("*.json"))
    print(f"🔍 Scanning {len(json_files)} annotation files for speed-limit signs...")
    
    for json_file in tqdm(json_files, desc="Scanning annotations"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            matching_labels = []
            
            # Check if this annotation has objects with speed-limit labels
            if 'objects' in data:
                for obj in data['objects']:
                    if 'label' in obj and pattern_regex.search(obj['label']):
                        matching_labels.append(obj['label'])
            
            if matching_labels:
                speed_limit_files[json_file.name] = matching_labels
                
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"⚠️  Warning: Could not process {json_file.name}: {e}")
            continue
    
    return speed_limit_files


def find_corresponding_images(annotation_filename: str, images_dir: Path) -> List[Path]:
    """
    Find image files corresponding to an annotation file.
    
    Args:
        annotation_filename: Name of the annotation JSON file
        images_dir: Path to the images directory
    
    Returns:
        List of corresponding image file paths
    """
    # Remove .json extension to get base name
    base_name = annotation_filename.replace('.json', '')
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    corresponding_images = []
    
    for ext in image_extensions:
        image_path = images_dir / f"{base_name}{ext}"
        if image_path.exists():
            corresponding_images.append(image_path)
    
    # Also check in subdirectories if images are organized that way
    if not corresponding_images:
        for subdir in images_dir.iterdir():
            if subdir.is_dir():
                for ext in image_extensions:
                    image_path = subdir / f"{base_name}{ext}"
                    if image_path.exists():
                        corresponding_images.append(image_path)
    
    return corresponding_images


def create_speed_signs_directory(base_dir: Path, dir_name: str = "speed_signs_extracted") -> Path:
    """
    Create the directory structure for speed limit signs.
    
    Args:
        base_dir: Base directory where to create the speed signs directory
        dir_name: Name of the directory to create
    
    Returns:
        Path to the created speed signs directory
    """
    speed_signs_dir = base_dir / dir_name
    annotations_dir = speed_signs_dir / "annotations"
    images_dir = speed_signs_dir / "images"
    
    # Create directories
    annotations_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created speed signs directory: {speed_signs_dir}")
    print(f"   • Annotations: {annotations_dir}")
    print(f"   • Images: {images_dir}")
    
    return speed_signs_dir


def move_files(speed_limit_files: Dict[str, List[str]], 
               annotations_dir: Path, 
               images_dir: Path, 
               speed_signs_dir: Path,
               copy_mode: bool = False) -> Dict[str, Any]:
    """
    Move (or copy) annotation files and corresponding images to speed signs directory.
    
    Args:
        speed_limit_files: Dictionary of annotation files with speed limit signs
        annotations_dir: Source annotations directory
        images_dir: Source images directory  
        speed_signs_dir: Destination speed signs directory
        copy_mode: If True, copy files instead of moving them
    
    Returns:
        Dictionary with statistics about the operation
    """
    stats = {
        'annotations_moved': 0,
        'images_moved': 0,
        'failed_annotations': 0,
        'failed_images': 0,
        'missing_images': []
    }
    
    dest_annotations_dir = speed_signs_dir / "annotations"
    dest_images_dir = speed_signs_dir / "images"
    
    operation = "Copying" if copy_mode else "Moving"
    print(f"\n📦 {operation} {len(speed_limit_files)} annotation files and their corresponding images...")
    
    for annotation_file, labels in tqdm(speed_limit_files.items(), desc=f"{operation} files"):
        # Move/copy annotation file
        src_annotation = annotations_dir / annotation_file
        dest_annotation = dest_annotations_dir / annotation_file
        
        try:
            if copy_mode:
                shutil.copy2(src_annotation, dest_annotation)
            else:
                shutil.move(str(src_annotation), str(dest_annotation))
            stats['annotations_moved'] += 1
            
            # Find and move/copy corresponding images
            corresponding_images = find_corresponding_images(annotation_file, images_dir)
            
            if not corresponding_images:
                stats['missing_images'].append(annotation_file)
                print(f"⚠️  Warning: No corresponding image found for {annotation_file}")
            
            for image_path in corresponding_images:
                dest_image = dest_images_dir / image_path.name
                try:
                    if copy_mode:
                        shutil.copy2(image_path, dest_image)
                    else:
                        shutil.move(str(image_path), str(dest_image))
                    stats['images_moved'] += 1
                except Exception as e:
                    print(f"❌ Failed to {operation.lower()} image {image_path.name}: {e}")
                    stats['failed_images'] += 1
            
        except Exception as e:
            print(f"❌ Failed to {operation.lower()} annotation {annotation_file}: {e}")
            stats['failed_annotations'] += 1
    
    return stats


def create_summary_report(speed_limit_files: Dict[str, List[str]], 
                         stats: Dict[str, Any], 
                         speed_signs_dir: Path):
    """
    Create a summary report of the extraction process.
    
    Args:
        speed_limit_files: Dictionary of files with speed limit signs
        stats: Statistics from the move operation
        speed_signs_dir: Path to the speed signs directory
    """
    report_file = speed_signs_dir / "extraction_summary.txt"
    
    # Count total labels
    all_labels = []
    for labels in speed_limit_files.values():
        all_labels.extend(labels)
    
    # Count unique labels
    unique_labels = set(all_labels)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("SPEED LIMIT SIGNS EXTRACTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total annotation files processed: {stats['annotations_moved'] + stats['failed_annotations']}\n")
        f.write(f"Successfully moved annotations: {stats['annotations_moved']}\n")
        f.write(f"Failed annotation moves: {stats['failed_annotations']}\n")
        f.write(f"Successfully moved images: {stats['images_moved']}\n")
        f.write(f"Failed image moves: {stats['failed_images']}\n")
        f.write(f"Annotations without corresponding images: {len(stats['missing_images'])}\n\n")
        
        f.write(f"Total speed-limit signs found: {len(all_labels)}\n")
        f.write(f"Unique speed-limit sign types: {len(unique_labels)}\n\n")
        
        f.write("UNIQUE SPEED-LIMIT SIGN TYPES FOUND:\n")
        f.write("-" * 40 + "\n")
        for label in sorted(unique_labels):
            count = all_labels.count(label)
            f.write(f"• {label} ({count} instances)\n")
        
        if stats['missing_images']:
            f.write(f"\nANNOTATIONS WITHOUT CORRESPONDING IMAGES:\n")
            f.write("-" * 40 + "\n")
            for annotation in stats['missing_images']:
                f.write(f"• {annotation}\n")
    
    print(f"📄 Summary report saved to: {report_file}")


def main():
    """Main function to extract speed limit signs from MTSD dataset."""
    parser = argparse.ArgumentParser(
        description="Extract speed-limit labeled traffic signs from MTSD dataset"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data_extracted",
        help="Path to the extracted data directory (default: data_extracted)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="speed_signs_extracted", 
        help="Name of output directory for speed limit signs (default: speed_signs_extracted)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (keeps originals)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="speed-limit",
        help="Regex pattern to match in labels (default: speed-limit)"
    )
    
    args = parser.parse_args()
    
    print("🚦 Speed Limit Signs Extractor")
    print("=" * 50)
    
    # Set up paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / args.data_dir
    annotations_dir = data_dir / "mtsd_v2_fully_annotated" / "annotations" # data_extracted/mtsd_v2_fully_annotated/annotations
    images_dir = data_dir / "images" 
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Annotations directory: {annotations_dir}")
    print(f"📁 Images directory: {images_dir}")
    print(f"🔍 Search pattern: '{args.pattern}'")
    print(f"📦 Operation mode: {'Copy' if args.copy else 'Move'}")
    print()
    
    # Validate input directories
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("💡 Tip: Run download_and_extract_dataset.py first to download the MTSD dataset")
        return 1
    
    if not annotations_dir.exists():
        print(f"❌ Error: Annotations directory not found: {annotations_dir}")
        return 1
    
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return 1
    
    # Find speed limit annotations
    speed_limit_files = find_speed_limit_annotations(annotations_dir, args.pattern)
    
    if not speed_limit_files:
        print("❌ No speed-limit signs found in the dataset!")
        print(f"   Searched pattern: '{args.pattern}'")
        print("   Try using a different pattern with --pattern option")
        return 1
    
    print(f"\n✅ Found {len(speed_limit_files)} annotation files with speed-limit signs!")
    
    # Show some examples
    print("\n📋 Examples of speed-limit labels found:")
    example_count = 0
    for filename, labels in speed_limit_files.items():
        for label in labels:
            print(f"   • {label} (in {filename})")
            example_count += 1
            if example_count >= 5:  # Show first 5 examples
                break
        if example_count >= 5:
            break
    
    if len(speed_limit_files) > 5:
        print(f"   ... and {sum(len(labels) for labels in speed_limit_files.values()) - example_count} more")
    
    # Create speed signs directory
    speed_signs_dir = create_speed_signs_directory(base_dir, args.output_dir)
    
    # Move/copy files
    stats = move_files(
        speed_limit_files, 
        annotations_dir, 
        images_dir, 
        speed_signs_dir,
        copy_mode=args.copy
    )
    
    # Create summary report
    create_summary_report(speed_limit_files, stats, speed_signs_dir)
    
    # Final summary
    operation = "copied" if args.copy else "moved"
    print(f"\n🎉 Extraction completed!")
    print(f"📊 Summary:")
    print(f"   • Annotation files {operation}: {stats['annotations_moved']}")
    print(f"   • Image files {operation}: {stats['images_moved']}")
    print(f"   • Failed operations: {stats['failed_annotations'] + stats['failed_images']}")
    print(f"   • Missing images: {len(stats['missing_images'])}")
    print(f"   • Output directory: {speed_signs_dir}")
    
    if stats['failed_annotations'] > 0 or stats['failed_images'] > 0:
        print(f"\n⚠️  Warning: Some files failed to {operation.rstrip('d')}. Check the output above for details.")
        return 1
    
    print(f"\n✅ All speed-limit signs successfully extracted!")
    print(f"   You can now use the files in '{args.output_dir}' to train an AI model specifically on speed limit signs.")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)