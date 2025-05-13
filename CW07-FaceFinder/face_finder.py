`import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Globals
DEFAULT_MODEL_PATH = 'CW07-FaceFinder/model/face_classifier_final.h5'
DEFAULT_CLASSES_PATH = 'CW07-FaceFinder/model/classes.npy'
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_WINDOW_SIZE = "800x600"
FRAMERATE = 30  # fps
OTHERS_CONFIDENCE_THRESHOLD = 0.6  # threshold for "Others" class
TARGET_FACE_SIZE = (224, 224)
MIN_FACE_SIZE = (30, 30)

class FaceRecognizer:
    """Main class for face detection and recognition"""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH, 
                 confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load classifier model
        self.model = load_model(model_path)
        self.class_names = np.load(classes_path)
        self.confidence_threshold = confidence_threshold
        
        # Check if "Others" class exists, get its index
        self.others_idx = -1
        for i, name in enumerate(self.class_names):
            if name == "Others":
                self.others_idx = i
                break
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE
        )
        return faces
    
    def process_face(self, face_img, target_size=TARGET_FACE_SIZE):
        """Preprocess a face for the classifier"""
        # Resize
        face_resized = cv2.resize(face_img, target_size)
        
        # Convert to RGB (model was trained on RGB)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        face_normalized = face_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def classify_face(self, face_batch):
        """Two-step classification process"""
        # Make prediction
        predictions = self.model.predict(face_batch, verbose=0)[0]
        
        # First check if it's likely to be "Others" or a known person
        if self.others_idx >= 0:
            others_confidence = predictions[self.others_idx]
            
            # If high confidence in Others class, return as Unknown
            if others_confidence > OTHERS_CONFIDENCE_THRESHOLD:
                return "Unknown", others_confidence, predictions
        
        # Get the highest confidence prediction (excluding Others)
        class_indices = np.argsort(predictions)[::-1]  # Sort indices by confidence (descending)
        
        # Skip Others in sorted indices if it's the highest
        if self.others_idx >= 0 and class_indices[0] == self.others_idx:
            class_idx = class_indices[1]  # Take second best if Others is best
            confidence = predictions[class_idx]
            # Only classify as a known person if confidence is high enough
            if confidence < self.confidence_threshold:
                return "Unknown", confidence, predictions
        else:
            class_idx = class_indices[0]
            confidence = predictions[class_idx]
            # Standard threshold check
            if confidence < self.confidence_threshold:
                return "Unknown", confidence, predictions
        
        # Return the class name and confidence
        class_name = self.class_names[class_idx]
        return class_name, confidence, predictions

class FaceRecognitionApp:
    """Application for face recognition using webcam"""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, 
                 classes_path=DEFAULT_CLASSES_PATH):
        # Initialize recognizer
        self.recognizer = FaceRecognizer(model_path, classes_path)
        
        # Set up UI
        self.root = tk.Tk()
        self.root.title("Face Recognition")
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Camera frame
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls frame
        controls = tk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Start/Stop button
        self.camera_active = False
        self.btn_text = tk.StringVar(value="Start Camera")
        tk.Button(controls, textvariable=self.btn_text, 
                command=self.toggle_camera).pack(side=tk.LEFT, padx=5)
        
        # Threshold slider
        tk.Label(controls, text="Confidence:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=self.recognizer.confidence_threshold)
        threshold_slider = tk.Scale(controls, from_=0.5, to=0.95, resolution=0.05,
                                  orient=tk.HORIZONTAL, length=200,
                                  variable=self.threshold_var,
                                  command=self.update_threshold)
        threshold_slider.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status, 
                            bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Camera variables
        self.cap = None
        self.photo = None
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start the camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status.set("Error: Could not open camera")
            return
        
        self.camera_active = True
        self.btn_text.set("Stop Camera")
        self.status.set("Camera active")
        
        self.update_frame()
    
    def stop_camera(self):
        """Stop the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.camera_active = False
        self.btn_text.set("Start Camera")
        self.status.set("Camera stopped")
        
        # Clear display
        self.canvas.delete("all")
    
    def update_threshold(self, *args):
        """Update confidence threshold"""
        self.recognizer.confidence_threshold = self.threshold_var.get()
    
    def update_frame(self):
        """Update camera frame and perform face recognition"""
        if not self.camera_active:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.status.set("Error: Failed to capture frame")
            self.stop_camera()
            return
        
        # Detect faces
        faces = self.recognizer.detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face
            face = frame[y:y+h, x:x+w]
            
            # Process and classify
            face_batch = self.recognizer.process_face(face)
            class_name, confidence, _ = self.recognizer.classify_face(face_batch)
            
            # Determine color based on confidence
            if confidence > 0.9:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.75:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Maintain aspect ratio
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            
            if canvas_height * aspect_ratio <= canvas_width:
                new_height = canvas_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo, anchor=tk.CENTER
        )
        
        # Update frame count in status
        face_count = len(faces)
        self.status.set(f"Detected {face_count} face{'s' if face_count != 1 else ''}")
        
        # Schedule next update - convert fps to milliseconds
        self.root.after(int(1000 / FRAMERATE), self.update_frame)
    
    def on_close(self):
        """Handle window close event"""
        self.stop_camera()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()`