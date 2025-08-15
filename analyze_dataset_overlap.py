#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze dataset overlap in data_raw/ by comparing file content hashes.
Shows how many files are common between different datasets before deduplication.

Usage:
    python analyze_dataset_overlap.py --raw_dir ../data_raw
"""

import argparse
import hashlib
import os
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(fname: str) -> bool:
    """Check if file is an image based on extension."""
    return os.path.splitext(fname.lower())[1] in IMG_EXTS

def file_hash(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """Calculate MD5 hash of file content."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[WARN] Failed to hash {path}: {e}")
        return None

def gather_images_recursive(start_dir: str) -> List[str]:
    """Recursively find all image files in directory."""
    files = []
    if not os.path.isdir(start_dir):
        return files
    
    for root, _, filenames in os.walk(start_dir):
        for filename in filenames:
            if is_image(filename):
                files.append(os.path.join(root, filename))
    return sorted(files)

def collect_dataset_paths(raw_dir: str) -> Dict[str, List[str]]:
    """Collect image paths for each dataset."""
    paths = {
        "HAM1000": [],
        "ISIC2019": [],
        "ISIC2020": [],
        "MIL10K_ISIC": [],
        "MIL10K_IL": [],
        "ITOBOS2024": []
    }

    # HAM1000
    ham = os.path.join(raw_dir, "HAM1000")
    ham_subs = [
        os.path.join(ham, "HAM10000_images_part_1"),
        os.path.join(ham, "HAM10000_images_part_2"),
        os.path.join(ham, "ISIC2018_Task3_Test_Images"),
    ]
    if os.path.isdir(ham):
        any_found = False
        for sd in ham_subs:
            if os.path.isdir(sd):
                paths["HAM1000"].extend(gather_images_recursive(sd))
                any_found = True
        if not any_found:
            paths["HAM1000"].extend(gather_images_recursive(ham))

    # ISIC2019
    is19 = os.path.join(raw_dir, "ISIC2019", "ISIC_2019_Training_Input")
    if os.path.isdir(is19):
        paths["ISIC2019"].extend(gather_images_recursive(is19))

    # ISIC2020
    is20 = os.path.join(raw_dir, "ISIC2020", "ISIC_2020_Train_Input")
    if os.path.isdir(is20):
        paths["ISIC2020"].extend(gather_images_recursive(is20))

    # MIL10K
    mil_root = os.path.join(raw_dir, "MILK10K")
    mil_images = os.path.join(mil_root, "images")
    mil_il = os.path.join(mil_root, "MILK10k_Training_Input")
    if os.path.isdir(mil_images):
        paths["MIL10K_ISIC"].extend(gather_images_recursive(mil_images))
    if os.path.isdir(mil_il):
        paths["MIL10K_IL"].extend(gather_images_recursive(mil_il))

    # ITOBOS2024
    itobos_imgs = os.path.join(raw_dir, "ITOBOS2024", "train", "images")
    if os.path.isdir(itobos_imgs):
        paths["ITOBOS2024"].extend(gather_images_recursive(itobos_imgs))

    return paths

def analyze_overlap(raw_dir: str):
    """Analyze file overlap between datasets."""
    print("🔍 Analyzing dataset overlap in:", raw_dir)
    print("=" * 60)
    
    # Collect paths
    paths = collect_dataset_paths(raw_dir)
    
    # Show file counts per dataset
    print("\n📊 Files found per dataset:")
    for dataset, file_list in paths.items():
        print(f"  {dataset:12}: {len(file_list):6} files")
    
    # Calculate hashes for each dataset
    print("\n🔄 Calculating file hashes...")
    dataset_hashes = {}
    hash_to_datasets = defaultdict(set)
    
    for dataset, file_list in paths.items():
        if not file_list:
            continue
            
        print(f"  Processing {dataset}...")
        dataset_hashes[dataset] = {}
        
        for file_path in file_list:
            file_hash_val = file_hash(file_path)
            if file_hash_val:
                dataset_hashes[dataset][file_path] = file_hash_val
                hash_to_datasets[file_hash_val].add(dataset)
    
    # Analyze overlaps
    print("\n📈 Overlap Analysis:")
    print("=" * 60)
    
    # Count files per hash (how many datasets share each file)
    hash_counts = Counter(len(datasets) for datasets in hash_to_datasets.values())
    
    print(f"\n📁 Files by number of datasets they appear in:")
    for count in sorted(hash_counts.keys()):
        num_files = hash_counts[count]
        percentage = (num_files / sum(hash_counts.values())) * 100
        print(f"  {count:2} datasets: {num_files:6} files ({percentage:5.1f}%)")
    
    # Show unique files per dataset
    print(f"\n🔒 Unique files per dataset (not shared with others):")
    for dataset in paths.keys():
        if dataset not in dataset_hashes:
            continue
        unique_files = sum(1 for hash_val in dataset_hashes[dataset].values() 
                          if len(hash_to_datasets[hash_val]) == 1)
        total_files = len(dataset_hashes[dataset])
        unique_pct = (unique_files / total_files * 100) if total_files > 0 else 0
        print(f"  {dataset:12}: {unique_files:6}/{total_files:6} unique ({unique_pct:5.1f}%)")
    
    # Show pairwise overlaps
    print(f"\n🔗 Pairwise overlaps (files shared between exactly 2 datasets):")
    dataset_names = list(paths.keys())
    
    for i, dataset1 in enumerate(dataset_names):
        for dataset2 in dataset_names[i+1:]:
            if dataset1 not in dataset_hashes or dataset2 not in dataset_hashes:
                continue
                
            # Count files shared between exactly these two datasets
            shared_count = 0
            for hash_val, datasets in hash_to_datasets.items():
                if datasets == {dataset1, dataset2}:
                    shared_count += 1
            
            if shared_count > 0:
                print(f"  {dataset1:12} ↔ {dataset2:12}: {shared_count:6} files")
    
    # Show files shared by 3+ datasets
    print(f"\n🔗 Files shared by 3+ datasets:")
    for count in sorted(hash_counts.keys()):
        if count >= 3:
            files = [hash_val for hash_val, datasets in hash_to_datasets.items() 
                    if len(datasets) == count]
            print(f"  {count} datasets: {len(files)} files")
            
            # Show details for first few
            for i, hash_val in enumerate(files[:5]):
                datasets = hash_to_datasets[hash_val]
                print(f"    Hash {hash_val[:8]}... appears in: {', '.join(sorted(datasets))}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
    
    # Summary statistics
    total_unique_files = len(hash_to_datasets)
    total_files_with_duplicates = sum(hash_counts.values()) - hash_counts[1]
    
    print(f"\n📋 Summary:")
    print(f"  Total unique files: {total_unique_files}")
    print(f"  Files with duplicates: {total_files_with_duplicates}")
    print(f"  Duplication rate: {(total_files_with_duplicates / total_unique_files * 100):.1f}%")
    
    # Show some example hashes for files shared by multiple datasets
    print(f"\n🔍 Example files shared by multiple datasets:")
    for count in sorted(hash_counts.keys()):
        if count >= 2:
            files = [hash_val for hash_val, datasets in hash_to_datasets.items() 
                    if len(datasets) == count]
            print(f"  {count} datasets: {len(files)} files")
            
            # Show details for first few
            for i, hash_val in enumerate(files[:3]):
                datasets = hash_to_datasets[hash_val]
                print(f"    Hash {hash_val[:8]}... appears in: {', '.join(sorted(datasets))}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset overlap in data_raw")
    parser.add_argument("--raw_dir", type=str, default="../data_raw", 
                       help="Path to raw datasets (default: ../data_raw)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.raw_dir):
        print(f"❌ Error: Directory {args.raw_dir} not found!")
        return
    
    analyze_overlap(args.raw_dir)

if __name__ == "__main__":
    main()
