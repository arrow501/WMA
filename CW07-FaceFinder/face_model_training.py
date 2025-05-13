import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2

# Define the directories
base_dir = 'CW07-FaceFinder/faces'
output_dir = 'CW07-FaceFinder/faces_processed'
model_save_dir = 'CW07-FaceFinder/model'
classes = ['Arrow', 'Hanka', 'Inni', 'Johhny', 'Miki']

# Create directories if they don't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Define target size for normalization
target_size = (224, 224)  # Better for MobileNetV2

def preprocess_images():
    """Normalize and flip images, saving to output directory"""
    print("Processing and flipping images...")
    
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
            
            # Convert BGR to RGB (Keras uses RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize the image (resize)
            normalized_img = cv2.resize(img, target_size)
            
            # Save the normalized version with sequential naming
            norm_filename = f"{class_name}_{i:03d}.jpg"
            norm_path = os.path.join(output_class_dir, norm_filename)
            cv2.imwrite(norm_path, cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR))
            
            # Create and save a flipped version
            flipped_img = cv2.flip(normalized_img, 1)  # 1 for horizontal flip
            flip_filename = f"{class_name}_{i:03d}_flipped.jpg"
            flip_path = os.path.join(output_class_dir, flip_filename)
            cv2.imwrite(flip_path, cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR))

def load_dataset():
    """Load the normalized and flipped images into X and y arrays"""
    images = []
    labels = []
    
    print("Loading dataset...")
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(output_dir, class_name)
        print(f"Loading class {i}: {class_name}")
        
        # Get all image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc=class_name):
            img_path = os.path.join(class_dir, img_file)
            
            # Read image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(class_name)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save label encoder for inference
    np.save(os.path.join(model_save_dir, 'classes.npy'), label_encoder.classes_)
    
    # Convert to categorical
    y_categorical = to_categorical(y_encoded)
    
    return X, y_categorical, label_encoder.classes_

def create_model(num_classes):
    """Create and compile CNN model"""
    # Use MobileNetV2 as base model (lightweight and efficient)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Main function to preprocess data and train the model"""
    # Check if processed images exist, if not, create them
    if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
        preprocess_images()
    
    # Load dataset
    X, y, class_names = load_dataset()
    print(f"Dataset loaded: {X.shape[0]} images, {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    
    # Create and compile model
    model = create_model(len(class_names))
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        os.path.join(model_save_dir, 'face_classifier.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'training_history.png'))
    plt.show()
    
    # Save model
    model.save(os.path.join(model_save_dir, 'face_classifier_final.h5'))
    print(f"Model saved to {os.path.join(model_save_dir, 'face_classifier_final.h5')}")

if __name__ == "__main__":
    train_model()