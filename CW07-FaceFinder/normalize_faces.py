import os
import cv2
import numpy as np
from tqdm import tqdm

# Globals 
BASE_DIR = 'CW07-FaceFinder/faces'
OUTPUT_DIR = 'CW07-FaceFinder/faces_normalized' 
CLASSES = ['Arrow', 'Hanka', 'Johhny', 'Miki', 'Others']
NORMALIZED_SIZE = (400, 400) #px

def process_images():
    print("Processing images...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for class_name in CLASSES:
        class_dir = os.path.join(BASE_DIR, class_name)
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        
        # Create class directory in output if it doesn't exist
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        print(f"Processing class: {class_name}")
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each image (no flipping needed)
        for i, img_file in enumerate(tqdm(image_files, desc=class_name)):
            img_path = os.path.join(class_dir, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # Normalize the image
            normalized_img = cv2.resize(img, NORMALIZED_SIZE)
            
            # Save the normalized version with sequential naming
            norm_filename = f"{class_name}_{i:03d}.jpg"
            norm_path = os.path.join(output_class_dir, norm_filename)
            cv2.imwrite(norm_path, normalized_img)

def count_images():
    print("\nCounting images in each class:")
    for class_name in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"{class_name}: 0 images (directory doesn't exist yet)")
            continue
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{class_name}: {len(image_files)} images")
        
        # Check if we have the minimum 100 images required
        if len(image_files) < 100:
            print(f"  Warning: {class_name} has fewer than 100 images")

if __name__ == "__main__":
    # Count original images
    print("Original images:")
    for class_name in CLASSES:
        class_dir = os.path.join(BASE_DIR, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"{class_name}: {len(image_files)} images")
        else:
            print(f"{class_name}: Directory not found")
    
    # Process images
    process_images()
    
    # Count processed images
    print("\nAfter processing:")
    count_images()
    
    print("\nNormalization complete!")
    print(f"Processed images saved to: {OUTPUT_DIR}")
    print("Original images are preserved.")