#!/usr/bin/env python3
"""
Dataset Analysis Script for Class Distribution
Analyzes the distribution of classes in the YOLO dataset to identify potential imbalances
"""

import os
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

def load_class_names(labels_file):
    """Load class names from labels.txt"""
    with open(labels_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_labels_directory(labels_dir, class_names):
    """Analyze all label files in the Labels directory"""
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    class_counts = defaultdict(int)
    total_objects = 0
    files_with_classes = defaultdict(int)
    
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        file_classes = set()
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:  # YOLO format: class x y w h
                    class_id = int(parts[0])
                    if 0 <= class_id < len(class_names):
                        class_counts[class_id] += 1
                        file_classes.add(class_id)
                        total_objects += 1
        
        # Count files that contain each class
        for class_id in file_classes:
            files_with_classes[class_id] += 1
    
    return class_counts, total_objects, files_with_classes, len(label_files)

def print_analysis(class_counts, total_objects, files_with_classes, total_files, class_names):
    """Print detailed analysis of the dataset"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    print(f"Total label files: {total_files}")
    print(f"Total objects: {total_objects}")
    print(f"Average objects per image: {total_objects/total_files:.2f}")
    
    print("\nCLASS DISTRIBUTION:")
    print("-" * 60)
    print(f"{'Class ID':<8} {'Class Name':<20} {'Count':<8} {'Percentage':<12} {'Files':<8}")
    print("-" * 60)
    
    # Sort by class ID
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_objects) * 100
        files_count = files_with_classes[class_id]
        class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
        
        print(f"{class_id:<8} {class_name:<20} {count:<8} {percentage:<12.2f}% {files_count:<8}")
    
    # Identify potential class imbalances
    print("\nCLASS IMBALANCE ANALYSIS:")
    print("-" * 40)
    
    counts = list(class_counts.values())
    if counts:
        min_count = min(counts)
        max_count = max(counts)
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"Min class count: {min_count}")
        print(f"Max class count: {max_count}")
        print(f"Imbalance ratio: {ratio:.2f}:1")
        
        if ratio > 10:
            print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        elif ratio > 5:
            print("⚠️  Moderate class imbalance detected")
        else:
            print("✅ Class distribution is relatively balanced")
    
    # Identify confused classes specifically
    print("\nCONFUSED CLASSES ANALYSIS:")
    print("-" * 40)
    
    confused_pairs = [
        ("black loungers", "red loungers"),
        ("gray bins", "concrete benches")
    ]
    
    for class1_name, class2_name in confused_pairs:
        class1_id = None
        class2_id = None
        
        # Find class IDs by name (case-insensitive partial match)
        for i, name in enumerate(class_names):
            if "black" in name.lower() and "loung" in name.lower():
                class1_id = i
            elif "red" in name.lower() and "loung" in name.lower():
                class2_id = i
            elif "gray" in name.lower() and "bin" in name.lower():
                class1_id = i
            elif "concrete" in name.lower() and "bench" in name.lower():
                class2_id = i
        
        if class1_id is not None and class2_id is not None:
            count1 = class_counts.get(class1_id, 0)
            count2 = class_counts.get(class2_id, 0)
            
            print(f"{class1_name} (ID {class1_id}): {count1} objects")
            print(f"{class2_name} (ID {class2_id}): {count2} objects")
            
            if count1 > 0 and count2 > 0:
                ratio = max(count1, count2) / min(count1, count2)
                print(f"Imbalance ratio: {ratio:.2f}:1")
                
                if ratio > 3:
                    print("⚠️  These confused classes have significant imbalance!")
                else:
                    print("✅ These confused classes are relatively balanced")
            print()

def create_visualization(class_counts, class_names):
    """Create visualizations of class distribution"""
    if not class_counts:
        print("No data to visualize")
        return
    
    # Prepare data for plotting
    class_ids = sorted(class_counts.keys())
    counts = [class_counts[cid] for cid in class_ids]
    labels = [class_names[cid] if cid < len(class_names) else f"Class_{cid}" for cid in class_ids]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    bars = ax1.bar(range(len(class_ids)), counts, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Number of Objects')
    ax1.set_title('Class Distribution - Object Counts')
    ax1.set_xticks(range(len(class_ids)))
    ax1.set_xticklabels([str(cid) for cid in class_ids])
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=[f"{cid}: {label[:15]}" for cid, label in zip(class_ids, labels)], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution - Percentages')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved as 'class_distribution.png'")
    
    return fig

def main():
    # File paths
    labels_dir = "Labels"
    labels_file = "labels.txt"
    
    # Check if files exist
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory '{labels_dir}' not found!")
        return
    
    if not os.path.exists(labels_file):
        print(f"Error: Labels file '{labels_file}' not found!")
        return
    
    # Load class names
    class_names = load_class_names(labels_file)
    print(f"Loaded {len(class_names)} class names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Analyze the dataset
    class_counts, total_objects, files_with_classes, total_files = analyze_labels_directory(labels_dir, class_names)
    
    # Print analysis
    print_analysis(class_counts, total_objects, files_with_classes, total_files, class_names)
    
    # Create visualization
    try:
        create_visualization(class_counts, class_names)
    except Exception as e:
        print(f"Could not create visualization: {e}")
        print("Install matplotlib to see visualizations: pip install matplotlib")

if __name__ == "__main__":
    main()
