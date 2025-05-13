import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

# Define the directories
base_dir = 'CW07-FaceFinder/faces'
output_dir = './faces_processed'  # New directory for processed images
classes = ['Arrow', 'Hanka', 'Inni', 'Johhny', 'Miki']

# Define target size for normalization
target_size = (400, 400)  # 400x400 as requested

def process_images():
    print("Processing and flipping images...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        # Create class directory in output if it doesn't exist
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        print(f"Processing class: {class_name}")
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each image and create a flipped version
        for i, img_file in enumerate(tqdm(image_files, desc=class_name)):
            img_path = os.path.join(class_dir, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # Normalize the image (resize to 400x400)
            normalized_img = cv2.resize(img, target_size)
            
            # Create sequentially named files
            # Save the normalized version with sequential naming
            norm_filename = f"{class_name}_{i:03d}.jpg"
            norm_path = os.path.join(output_class_dir, norm_filename)
            cv2.imwrite(norm_path, normalized_img)
            
            # Create and save a flipped version
            flipped_img = cv2.flip(normalized_img, 1)  # 1 for horizontal flip
            flip_filename = f"{class_name}_{i:03d}_flipped.jpg"
            flip_path = os.path.join(output_class_dir, flip_filename)
            cv2.imwrite(flip_path, flipped_img)

def count_images():
    print("\nCounting images in each class:")
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"{class_name}: 0 images (directory doesn't exist yet)")
            continue
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{class_name}: {len(image_files)} images")

if __name__ == "__main__":
    # Count original images
    print("Original images:")
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"{class_name}: {len(image_files)} images")
    
    # Process images
    process_images()
    
    # Count processed images
    print("\nAfter processing (normalized + flipped):")
    count_images()
    
    print("\nNormalization and flipping complete!")
    print(f"Processed images saved to: {output_dir}")
    print("Original images are preserved.")