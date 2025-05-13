import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Globals
# Directories
DATA_DIR = 'CW07-FaceFinder/faces_normalized'
OUTPUT_DIR = 'CW07-FaceFinder/model'

# Model parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.0001  # Slightly higher than before

# Training parameters
EPOCHS = 30
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 7  # Increased patience
FINE_TUNING_PATIENCE = 5  # Increased patience for fine-tuning
PHASE1_EPOCHS_RATIO = 0.5
PHASE2_EPOCHS_RATIO = 0.5  # Equal split between phases

# Regularization parameters
DROPOUT_RATES = (0.3, 0.2)  # Reduced dropout rates
FINE_TUNING_LAYERS = 25  # More layers to fine-tune

# Data augmentation parameters
ROTATION_RANGE = 15  # Reduced from 20
SHIFT_RANGE = 0.15  # Reduced from 0.2
SHEAR_RANGE = 0.15  # Reduced from 0.2
ZOOM_RANGE = 0.15  # Reduced from 0.2
BRIGHTNESS_RANGE = [0.85, 1.15]  # Less extreme


class FaceModelTrainer:
    """Class for training a face recognition model"""
    
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Image data generator for training augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=ROTATION_RANGE,
            width_shift_range=SHIFT_RANGE,
            height_shift_range=SHIFT_RANGE,
            shear_range=SHEAR_RANGE,
            zoom_range=ZOOM_RANGE,
            horizontal_flip=True,
            brightness_range=BRIGHTNESS_RANGE,
            fill_mode='nearest'
        )
        
        # Validation data generator (just rescaling)
        self.val_datagen = ImageDataGenerator(rescale=1./255)
    
    def load_data(self):
        """Load images from directories"""
        print("Loading data...")
        
        images = []
        labels = []
        self.classes = []
        
        # Get class directories
        class_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        self.classes = sorted(class_dirs)
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            print(f"Loading class {class_idx}: {class_name}")
            
            # Get image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Load images
            for img_file in tqdm(image_files):
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                # Convert to RGB and resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                
                images.append(img)
                labels.append(class_name)
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Save classes for inference
        np.save(os.path.join(self.output_dir, 'classes.npy'), self.label_encoder.classes_)
        
        # Convert to categorical
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=VALIDATION_SPLIT, stratify=y_encoded, random_state=RANDOM_STATE
        )
        
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Validation set: {X_val.shape[0]} images")
        
        # Compute class weights (for handling imbalance)
        class_counts = np.bincount(y_encoded)
        total = np.sum(class_counts)
        
        # Apply square root to smooth out extreme weights
        # This helps prevent catastrophic forgetting for minority classes
        # while not over-penalizing the majority class
        smoothed_counts = np.sqrt(class_counts)
        self.class_weights = {i: total / (len(class_counts) * smoothed_counts[i] * np.sum(smoothed_counts) / total) 
                             for i, count in enumerate(class_counts)}
        
        # Print class weights
        print("Class weights:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: {self.class_weights[i]:.4f}")
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self, num_classes):
        """Create model architecture"""
        # Use MobileNetV2 as base model
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # First, freeze all layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create model with less aggressive regularization
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(DROPOUT_RATES[0]),  # Reduced dropout
            layers.Dense(128, activation='relu'),
            layers.Dropout(DROPOUT_RATES[1]),  # Reduced dropout
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, epochs=EPOCHS):
        """Train the model with balanced regularization approach"""
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        # Create model
        model = self.create_model(len(self.classes))
        model.summary()
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Save best model during each phase
        checkpoint = ModelCheckpoint(
            os.path.join(self.output_dir, 'face_classifier.h5'),
            monitor='val_accuracy', 
            save_best_only=True,
            verbose=1
        )
        
        # Phase 1: Train with frozen base model
        print("Phase 1: Training with frozen base model...")
        history1 = model.fit(
            self.train_datagen.flow(X_train, y_train, batch_size=self.batch_size),
            validation_data=self.val_datagen.flow(X_val, y_val, batch_size=self.batch_size),
            epochs=int(epochs * PHASE1_EPOCHS_RATIO),
            callbacks=[early_stopping, checkpoint],
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Save best model from phase 1
        model.save(os.path.join(self.output_dir, 'phase1_model.h5'))
        
        # Phase 2: Fine-tuning with balanced regularization
        print("Phase 2: Fine-tuning with balanced regularization...")
        
        # Unfreeze more layers (25 instead of 15)
        for layer in model.layers[0].layers[-FINE_TUNING_LAYERS:]:
            layer.trainable = True
        
        # Recompile with moderate learning rate
        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Use moderate early stopping in phase 2
        early_stopping.patience = FINE_TUNING_PATIENCE
        
        history2 = model.fit(
            self.train_datagen.flow(X_train, y_train, batch_size=self.batch_size),
            validation_data=self.val_datagen.flow(X_val, y_val, batch_size=self.batch_size),
            epochs=int(epochs * PHASE2_EPOCHS_RATIO),
            callbacks=[early_stopping, checkpoint],
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Load the best model (may be from phase 1 if phase 2 didn't improve)
        from tensorflow.keras.models import load_model
        best_model = load_model(os.path.join(self.output_dir, 'face_classifier.h5'))
        
        # Final save
        best_model.save(os.path.join(self.output_dir, 'face_classifier_final.h5'))
        
        # Combine histories for plotting
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        # Plot training history
        self.plot_history(combined_history)
        
        return best_model
    
    def plot_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.show()

if __name__ == "__main__":
    trainer = FaceModelTrainer()
    model = trainer.train()