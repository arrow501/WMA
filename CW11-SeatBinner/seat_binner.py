#!/usr/bin/env python3
"""
Seat Binner GUI - YOLO Object Detection Interface
Real-time detection of seats and bins with video playback
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
import threading
import time
from ultralytics import YOLO
import os

class SeatBinnerGUI:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("Seat Binner - Object Detection")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Class colors and names
        self.class_names = ["black_lounger", "red_lounger", "concrete_bench", "gray_bin"]
        self.class_colors = {
            0: (0, 0, 0),        # black_lounger - black
            1: (255, 0, 0),      # red_lounger - red  
            2: (128, 128, 128),  # concrete_bench - gray
            3: (124, 252, 0)     # gray_bin - puke green
        }
        
        # Video and model variables
        self.current_video = None
        self.video_frames = []
        self.current_frame_idx = 0
        self.is_playing = False
        self.fps = 30
        self.model = None
        
        # Threading
        self.play_thread = None
        self.stop_playback = False
        
        # Initialize UI
        self.setup_ui()
        self.load_available_videos()
        self.load_available_models()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Header frame
        header_frame = ctk.CTkFrame(self.root)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_columnconfigure(3, weight=1)
        
        # Video selection
        ctk.CTkLabel(header_frame, text="Video:").grid(row=0, column=0, padx=10, pady=10)
        self.video_var = tk.StringVar()
        self.video_dropdown = ctk.CTkComboBox(header_frame, variable=self.video_var, 
                                            command=self.on_video_change, width=200)
        self.video_dropdown.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Model selection  
        ctk.CTkLabel(header_frame, text="Model:").grid(row=0, column=2, padx=10, pady=10)
        self.model_var = tk.StringVar()
        self.model_dropdown = ctk.CTkComboBox(header_frame, variable=self.model_var,
                                            command=self.on_model_change, width=200)
        self.model_dropdown.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        
        # Load model button
        self.load_btn = ctk.CTkButton(header_frame, text="Load Model", 
                                     command=self.load_model, width=100)
        self.load_btn.grid(row=0, column=4, padx=10, pady=10)
        
        # Main content frame
        content_frame = ctk.CTkFrame(self.root)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Video display frame
        self.video_frame = ctk.CTkFrame(content_frame)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.video_frame, bg='black', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Control panel
        control_frame = ctk.CTkFrame(self.root)
        control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Playback controls frame
        playback_frame = ctk.CTkFrame(control_frame)
        playback_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Play/Pause button
        self.play_btn = ctk.CTkButton(playback_frame, text="▶ Play", 
                                     command=self.toggle_playback, width=80)
        self.play_btn.grid(row=0, column=0, padx=5, pady=5)
        
        # Stop button  
        self.stop_btn = ctk.CTkButton(playback_frame, text="⏹ Stop",
                                     command=self.stop_video, width=80)
        self.stop_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Frame info
        self.frame_info = ctk.CTkLabel(playback_frame, text="Frame: 0/0")
        self.frame_info.grid(row=0, column=2, padx=20, pady=5)
        
        # Video scrubber
        scrubber_frame = ctk.CTkFrame(control_frame)
        scrubber_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        scrubber_frame.grid_columnconfigure(0, weight=1)
        
        self.scrubber = ctk.CTkSlider(scrubber_frame, from_=0, to=100, 
                                     command=self.on_scrubber_change)
        self.scrubber.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.scrubber.set(0)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_frame = ctk.CTkFrame(self.root)
        status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Legend frame
        legend_frame = ctk.CTkFrame(status_frame)
        legend_frame.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        
        self.create_legend(legend_frame)
        
    def create_legend(self, parent):
        """Create color legend for object classes"""
        ctk.CTkLabel(parent, text="Legend:", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=5, pady=2)
        
        for i, (class_id, color) in enumerate(self.class_colors.items()):
            # Create colored rectangle
            canvas = tk.Canvas(parent, width=15, height=15, highlightthickness=0)
            canvas.grid(row=0, column=i*2+1, padx=2, pady=2)
            
            # Convert RGB to hex
            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            canvas.create_rectangle(2, 2, 13, 13, fill=hex_color, outline="white")
            
            # Class name
            name = self.class_names[class_id].replace('_', ' ').title()
            ctk.CTkLabel(parent, text=name, font=ctk.CTkFont(size=10)).grid(
                row=0, column=i*2+2, padx=2, pady=2)
        
    def load_available_videos(self):
        """Load available videos from Video folder"""
        video_dir = Path("Video")
        if not video_dir.exists():
            video_dir.mkdir()
            
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(video_dir.glob(ext))
            
        video_names = [f.name for f in video_files]
        
        if video_names:
            self.video_dropdown.configure(values=video_names)
            self.video_var.set(video_names[0])
        else:
            self.video_dropdown.configure(values=["No videos found"])
            
    def load_available_models(self):
        """Load available models from Models folder"""
        models_dir = Path("Models")
        if not models_dir.exists():
            models_dir.mkdir()
            
        model_files = []
        for ext in ['*.pt', '*.onnx']:
            model_files.extend(models_dir.glob(ext))
            
        model_names = [f.name for f in model_files]
        
        if model_names:
            self.model_dropdown.configure(values=model_names)
            self.model_var.set(model_names[0])
        else:
            self.model_dropdown.configure(values=["No models found"])
            
    def on_video_change(self, choice):
        """Handle video selection change"""
        self.load_video(choice)
        
    def on_model_change(self, choice):
        """Handle model selection change"""
        self.status_var.set(f"Selected model: {choice}")
        
    def load_video(self, video_name):
        """Load and parse video into frames"""
        if video_name == "No videos found":
            return
            
        video_path = Path("Video") / video_name
        if not video_path.exists():
            messagebox.showerror("Error", f"Video file not found: {video_path}")
            return
            
        self.status_var.set(f"Loading video: {video_name}...")
        self.root.update()
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            self.fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            cap.release()
            
            self.video_frames = frames
            self.current_frame_idx = 0
            self.current_video = video_name
            
            # Update scrubber
            if len(frames) > 0:
                self.scrubber.configure(to=len(frames)-1)
                self.scrubber.set(0)
                self.update_frame_display()
                
            self.status_var.set(f"Loaded {len(frames)} frames from {video_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            self.status_var.set("Error loading video")
            
    def load_model(self):
        """Load selected YOLO model"""
        model_name = self.model_var.get()
        if model_name == "No models found":
            messagebox.showwarning("Warning", "No models available to load")
            return
            
        model_path = Path("Models") / model_name
        if not model_path.exists():
            messagebox.showerror("Error", f"Model file not found: {model_path}")
            return
            
        self.status_var.set(f"Loading model: {model_name}...")
        self.root.update()
        
        try:
            self.model = YOLO(str(model_path))
            self.status_var.set(f"Model loaded: {model_name}")
            messagebox.showinfo("Success", f"Model {model_name} loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Error loading model")
            
    def detect_objects(self, frame):
        """Run object detection on frame"""
        if self.model is None:
            return frame, []
            
        try:
            results = self.model(frame, verbose=False)
            return self.draw_detections(frame, results[0])
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []
            
    def draw_detections(self, frame, result):
        """Draw bounding boxes and labels on frame"""
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confidences, classes):
                if cls < len(self.class_names):
                    x1, y1, x2, y2 = box
                    color = self.class_colors.get(cls, (255, 255, 255))
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Prepare label
                    class_name = self.class_names[cls].replace('_', ' ').title()
                    label = f"{class_name} {conf:.2f}"
                    
                    # Add "sitable" badge for seats
                    if cls in [0, 1]:  # loungers
                        label += " 🪑"
                    
                    # Draw label background
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                        
                    bbox = draw.textbbox((x1, y1-25), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    
                    # Draw label text
                    text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
                    draw.text((x1, y1-25), label, fill=text_color, font=font)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        return np.array(img), detections
    
    def update_frame_display(self):
        """Update the video frame display"""
        if not self.video_frames or self.current_frame_idx >= len(self.video_frames):
            return
            
        frame = self.video_frames[self.current_frame_idx].copy()
        
        # Run detection if model is loaded
        if self.model is not None:
            frame, detections = self.detect_objects(frame)
        
        # Resize frame to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate aspect ratio preserving size
            frame_h, frame_w = frame.shape[:2]
            aspect_ratio = frame_w / frame_h
            
            if canvas_width / canvas_height > aspect_ratio:
                new_height = canvas_height
                new_width = int(canvas_height * aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(canvas_width / aspect_ratio)
            
            # Resize frame
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference
        
        # Update frame info
        total_frames = len(self.video_frames)
        self.frame_info.configure(text=f"Frame: {self.current_frame_idx + 1}/{total_frames}")
        
    def on_scrubber_change(self, value):
        """Handle scrubber position change"""
        if self.video_frames:
            self.current_frame_idx = int(value)
            self.update_frame_display()
            
    def toggle_playback(self):
        """Toggle video playback"""
        if not self.video_frames:
            messagebox.showwarning("Warning", "No video loaded")
            return
            
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
            
    def play_video(self):
        """Start video playback"""
        if self.is_playing:
            return
            
        self.is_playing = True
        self.stop_playback = False
        self.play_btn.configure(text="⏸ Pause")
        
        # Start playback thread
        self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.play_thread.start()
        
    def pause_video(self):
        """Pause video playback"""
        self.is_playing = False
        self.stop_playback = True
        self.play_btn.configure(text="▶ Play")
        
    def stop_video(self):
        """Stop video playback and reset to beginning"""
        self.pause_video()
        self.current_frame_idx = 0
        self.scrubber.set(0)
        self.update_frame_display()
        
    def _playback_loop(self):
        """Main playback loop (runs in separate thread)"""
        frame_time = 1.0 / self.fps
        
        while self.is_playing and not self.stop_playback:
            start_time = time.time()
            
            # Update frame
            self.root.after(0, self._advance_frame)
            
            # Wait for next frame
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
            
    def _advance_frame(self):
        """Advance to next frame (called from main thread)"""
        if not self.video_frames:
            return
            
        self.current_frame_idx += 1
        
        # Loop back to beginning if at end
        if self.current_frame_idx >= len(self.video_frames):
            self.current_frame_idx = 0
            
        # Update display
        self.scrubber.set(self.current_frame_idx)
        self.update_frame_display()
        
    def on_closing(self):
        """Handle application closing"""
        self.stop_playback = True
        self.is_playing = False
        
        # Wait for playback thread to finish
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
            
        self.root.destroy()
        
    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind resize event to update display
        self.canvas.bind('<Configure>', lambda e: self.root.after(100, self.update_frame_display))
        
        # Initial display update
        self.root.after(500, self.update_frame_display)
        
        self.root.mainloop()

def main():
    """Main entry point"""
    # Check for required directories
    required_dirs = ["Video", "Models", "Photos", "Labels"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"Creating directory: {dir_name}")
            dir_path.mkdir(exist_ok=True)
    
    # Check for labels.txt
    labels_file = Path("labels.txt")
    if not labels_file.exists():
        print("Creating default labels.txt file...")
        with open(labels_file, 'w') as f:
            f.write("black_lounger\nred_lounger\nconcrete_bench\ngray_bin\n")
    
    try:
        app = SeatBinnerGUI()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()