import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths
model_path = 'CW07-FaceFinder/model/face_classifier_final.h5'
classes_path = 'CW07-FaceFinder/model/classes.npy'

# Load model and class names
model = load_model(model_path)
class_names = np.load(classes_path)
print(f"Loaded model with classes: {class_names}")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess face for prediction
def preprocess_face(face_img):
    # Resize to match model input shape
    face_img = cv2.resize(face_img, (224, 224))
    # Convert to RGB (model was trained on RGB)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Normalize to [0,1]
    face_img = face_img.astype('float32') / 255.0
    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break
    
    # Make a copy for display
    display_frame = frame.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract face
        face = frame[y:y+h, x:x+w]
        
        # Preprocess face for prediction
        processed_face = preprocess_face(face)
        
        # Make prediction
        prediction = model.predict(processed_face, verbose=0)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx] * 100
        
        # Get class name
        class_name = class_names[class_idx]
        
        # Determine color based on confidence
        if confidence > 90:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 70:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw rectangle around face
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
        
        # Create label with class name and confidence
        label = f"{class_name}: {confidence:.1f}%"
        
        # Draw filled rectangle for text background
        cv2.rectangle(display_frame, (x, y-30), (x+w, y), color, -1)
        
        # Add text
        cv2.putText(display_frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Display frame
    cv2.imshow('Face Recognition', display_frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()