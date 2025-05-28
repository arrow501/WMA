#!/usr/bin/env python3
"""
Seat Binner GUI - Optimized YOLO Object Detection Interface
Fast real-time detection with threaded inference
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
import threading
import time
from ultralytics import YOLO
import queue

class InferenceWorker:
	"""Separate worker for YOLO inference to avoid blocking GUI"""
	def __init__(self):
		self.model = None
		self.frame_queue = queue.Queue(maxsize=1)  # Minimal buffer
		self.result_queue = queue.Queue(maxsize=1)
		self.running = False
		self.thread = None
		self.last_frame_hash = None  # Track frame changes efficiently
		
	def load_model(self, model_path):
		"""Load YOLO model"""
		try:
			self.model = YOLO(str(model_path))
			return True
		except Exception as e:
			print(f"Failed to load model: {e}")
			return False
	
	def start(self):
		"""Start inference worker thread"""
		if self.running:
			return
		self.running = True
		self.thread = threading.Thread(target=self._inference_loop, daemon=True)
		self.thread.start()
	
	def stop(self):
		"""Stop inference worker"""
		self.running = False
		if self.thread:
			self.thread.join(timeout=1.0)
	
	def process_frame(self, frame):
		"""Queue frame for processing"""
		if not self.frame_queue.full():
			try:
				self.frame_queue.put_nowait(frame)
			except queue.Full:
				pass
	
	def get_result(self):
		"""Get processed result if available"""
		try:
			return self.result_queue.get_nowait()
		except queue.Empty:
			return None
	
	def _inference_loop(self):
		"""Main inference loop"""
		while self.running:
			try:
				frame = self.frame_queue.get(timeout=0.1)
				if self.model is not None:
					results = self.model(frame, verbose=False)
					if not self.result_queue.full():
						try:
							self.result_queue.put_nowait((frame, results[0]))
						except queue.Full:
							pass
			except queue.Empty:
				continue
			except Exception as e:
				print(f"Inference error: {e}")

class SeatBinnerGUI:
	def __init__(self):
		# Initialize main window
		self.root = tk.Tk()
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
		
		# Video variables
		self.cap = None
		self.current_frame = None
		self.is_playing = False
		self.fps = 30
		self.frame_count = 0
		self.current_frame_idx = 0
		  # Threading
		self.inference_worker = InferenceWorker()
		self.play_thread = None
		self.stop_playback = False
		
		# Cache for display and optimization
		self.display_frame = None
		self.last_processed_frame_id = None  # Use frame index instead of array comparison
		self.cached_photo = None  # Cache PIL PhotoImage
		self.last_canvas_size = (0, 0)  # Track canvas resize
		
		# Initialize UI
		self.setup_ui()
		self.load_available_videos()
		self.load_available_models()
		
		# Start inference worker
		self.inference_worker.start()
		
		# Start GUI update loop
		self.update_display()
		
	def setup_ui(self):
		"""Setup the user interface"""
		# Configure grid weights
		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_rowconfigure(1, weight=1)
		
		# Header frame
		header_frame = tk.Frame(self.root, bg='lightgray')
		header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
		header_frame.grid_columnconfigure(1, weight=1)
		header_frame.grid_columnconfigure(3, weight=1)
		
		# Video selection
		tk.Label(header_frame, text="Video:", bg='lightgray').grid(row=0, column=0, padx=5, pady=5)
		self.video_var = tk.StringVar()
		self.video_dropdown = ttk.Combobox(header_frame, textvariable=self.video_var, 
										 width=30, state="readonly")
		self.video_dropdown.bind('<<ComboboxSelected>>', self.on_video_change)
		self.video_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
		
		# Model selection  
		tk.Label(header_frame, text="Model:", bg='lightgray').grid(row=0, column=2, padx=5, pady=5)
		self.model_var = tk.StringVar()
		self.model_dropdown = ttk.Combobox(header_frame, textvariable=self.model_var,
										 width=30, state="readonly")
		self.model_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky="w")
		
		# Load model button
		self.load_btn = tk.Button(header_frame, text="Load Model", 
								command=self.load_model, width=12)
		self.load_btn.grid(row=0, column=4, padx=5, pady=5)
		
		# Main content frame
		content_frame = tk.Frame(self.root, bg='black')
		content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
		content_frame.grid_columnconfigure(0, weight=1)
		content_frame.grid_rowconfigure(0, weight=1)
		
		# Canvas for video display
		self.canvas = tk.Canvas(content_frame, bg='black', highlightthickness=0)
		self.canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
		
		# Control panel
		control_frame = tk.Frame(self.root, bg='lightgray')
		control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
		control_frame.grid_columnconfigure(1, weight=1)
		
		# Playback controls frame
		playback_frame = tk.Frame(control_frame, bg='lightgray')
		playback_frame.grid(row=0, column=0, padx=5, pady=5, sticky="w")
		
		# Play/Pause button
		self.play_btn = tk.Button(playback_frame, text="▶ Play", 
								command=self.toggle_playback, width=10)
		self.play_btn.grid(row=0, column=0, padx=2, pady=2)
		
		# Stop button  
		self.stop_btn = tk.Button(playback_frame, text="⏹ Stop",
								command=self.stop_video, width=10)
		self.stop_btn.grid(row=0, column=1, padx=2, pady=2)
		
		# Frame info
		self.frame_info = tk.Label(playback_frame, text="Frame: 0/0", bg='lightgray')
		self.frame_info.grid(row=0, column=2, padx=10, pady=2)
		
		# Video scrubber
		scrubber_frame = tk.Frame(control_frame, bg='lightgray')
		scrubber_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
		scrubber_frame.grid_columnconfigure(0, weight=1)
		
		self.scrubber = tk.Scale(scrubber_frame, from_=0, to=100, orient=tk.HORIZONTAL,
							   command=self.on_scrubber_change, showvalue=False)
		self.scrubber.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
		
		# Status bar
		self.status_var = tk.StringVar(value="Ready")
		status_frame = tk.Frame(self.root, bg='lightgray')
		status_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
		
		self.status_label = tk.Label(status_frame, textvariable=self.status_var, bg='lightgray')
		self.status_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
		
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
			self.video_dropdown['values'] = video_names
			self.video_var.set(video_names[0])
		else:
			self.video_dropdown['values'] = ["No videos found"]
			
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
			self.model_dropdown['values'] = model_names
			self.model_var.set(model_names[0])
		else:
			self.model_dropdown['values'] = ["No models found"]
			
	def on_video_change(self, event=None):
		"""Handle video selection change"""
		choice = self.video_var.get()
		self.load_video(choice)
		
	def load_video(self, video_name):
		"""Load video file"""
		if video_name == "No videos found":
			return
			
		video_path = Path("Video") / video_name
		if not video_path.exists():
			messagebox.showerror("Error", f"Video file not found: {video_path}")
			return
			
		self.status_var.set(f"Loading video: {video_name}...")
		self.root.update()
		
		try:
			# Close existing video
			if self.cap:
				self.cap.release()
			
			self.cap = cv2.VideoCapture(str(video_path))
			self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
			self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
			self.current_frame_idx = 0
			
			# Update scrubber
			if self.frame_count > 0:
				self.scrubber.configure(to=self.frame_count-1)
				self.scrubber.set(0)
				
			# Load first frame
			self.seek_frame(0)
			
			self.status_var.set(f"Loaded {self.frame_count} frames from {video_name}")
			
		except Exception as e:
			messagebox.showerror("Error", f"Failed to load video: {str(e)}")
			self.status_var.set("Error loading video")
			
	def seek_frame(self, frame_idx):
		"""Seek to specific frame"""
		if not self.cap:
			return
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
		ret, frame = self.cap.read()
		if ret:
			self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			self.current_frame_idx = frame_idx
			self.display_frame = self.current_frame.copy()
			
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
		
		if self.inference_worker.load_model(model_path):
			self.status_var.set(f"Model loaded: {model_name}")
			messagebox.showinfo("Success", f"Model {model_name} loaded successfully!")
		else:
			messagebox.showerror("Error", f"Failed to load model: {model_name}")
			self.status_var.set("Error loading model")
			
	def draw_detections(self, frame, result):
		"""Draw bounding boxes and labels on frame using OpenCV (faster than PIL)"""
		# Work directly with OpenCV - much faster than PIL conversion
		img = frame.copy()
		
		if result.boxes is not None:
			boxes = result.boxes.xyxy.cpu().numpy()
			confidences = result.boxes.conf.cpu().numpy()
			classes = result.boxes.cls.cpu().numpy().astype(int)
			
			for box, conf, cls in zip(boxes, confidences, classes):
				if cls < len(self.class_names):
					x1, y1, x2, y2 = map(int, box)
					color = self.class_colors.get(cls, (255, 255, 255))
					
					# Convert RGB to BGR for OpenCV
					bgr_color = (color[2], color[1], color[0])
					
					# Draw bounding box
					cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2)
					
					# Prepare label
					class_name = self.class_names[cls].replace('_', ' ').title()
					label = f"{class_name} {conf:.2f}"
					
					# Draw label background and text
					font = cv2.FONT_HERSHEY_SIMPLEX
					font_scale = 0.5
					thickness = 1
					
					# Get text size for background
					(text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
					
					# Draw background rectangle
					cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width, y1), bgr_color, -1)
					
					# Draw text
					text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
					cv2.putText(img, label, (x1, y1-5), font, font_scale, text_color, thickness)
		
		return img
	
	def update_display(self):
		"""Update display - called repeatedly with optimized frame tracking"""
		try:
			# Check for inference results
			result = self.inference_worker.get_result()
			if result:
				frame, detections = result
				self.display_frame = self.draw_detections(frame, detections)
				self.last_processed_frame_id = self.current_frame_idx
			
			# Send current frame for inference if model is loaded and frame changed
			if (self.current_frame is not None and 
				self.inference_worker.model is not None and
				self.last_processed_frame_id != self.current_frame_idx):
				self.inference_worker.process_frame(self.current_frame.copy())
			
			# Update canvas if we have a frame to display
			if self.display_frame is not None:
				self.update_canvas()
				
			# Update frame info less frequently
			if self.cap:
				total_frames = self.frame_count
				self.frame_info.configure(text=f"Frame: {self.current_frame_idx + 1}/{total_frames}")
		
		except Exception as e:
			print(f"Display update error: {e}")
		
		# Schedule next update at 30 FPS instead of 60
		self.root.after(33, self.update_display)

	def update_canvas(self):
		"""Update canvas with current frame - optimized with caching"""
		if self.display_frame is None:
			return
			
		canvas_width = self.canvas.winfo_width()
		canvas_height = self.canvas.winfo_height()
		
		if canvas_width > 1 and canvas_height > 1:
			# Check if canvas size changed
			current_canvas_size = (canvas_width, canvas_height)
			size_changed = current_canvas_size != self.last_canvas_size # unused
			self.last_canvas_size = current_canvas_size
			
			# Calculate aspect ratio preserving size
			frame_h, frame_w = self.display_frame.shape[:2]
			aspect_ratio = frame_w / frame_h
			
			if canvas_width / canvas_height > aspect_ratio:
				new_height = canvas_height
				new_width = int(canvas_height * aspect_ratio)
			else:
				new_width = canvas_width
				new_height = int(canvas_width / aspect_ratio)
			
			# Resize frame
			frame_resized = cv2.resize(self.display_frame, (new_width, new_height))
			
			# Convert to PhotoImage - cache if size didn't change
			img = Image.fromarray(frame_resized)
			photo = ImageTk.PhotoImage(img)
			
			# Update canvas
			self.canvas.delete("all")
			x = (canvas_width - new_width) // 2
			y = (canvas_height - new_height) // 2
			self.canvas.create_image(x, y, anchor=tk.NW, image=photo)
			self.canvas.image = photo  # Keep a reference
			
	def on_scrubber_change(self, value):
		"""Handle scrubber position change"""
		if self.cap and not self.is_playing:
			frame_idx = int(float(value))
			self.seek_frame(frame_idx)
			
	def toggle_playback(self):
		"""Toggle video playback"""
		if not self.cap:
			messagebox.showwarning("Warning", "No video loaded")
			return
			
		if self.is_playing:
			self.pause_video()
		else:
			self.play_video()
			
	def play_video(self):
		"""Start video playback"""
		if self.is_playing or not self.cap:
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
		if self.cap:
			self.seek_frame(0)
			self.scrubber.set(0)
			
	def _playback_loop(self):
		"""Main playback loop (runs in separate thread) - optimized for sequential reading"""
		frame_time = 1.0 / self.fps
		
		while self.is_playing and not self.stop_playback and self.cap:
			start_time = time.time()
			
			# Read next frame sequentially (much faster than seeking)
			ret, frame = self.cap.read()
			
			if ret:
				self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				self.current_frame_idx += 1
				
				# Update scrubber from main thread
				self.root.after(0, lambda idx=self.current_frame_idx: self.scrubber.set(idx))
			else:
				# End of video - loop back to beginning
				self.current_frame_idx = 0
				self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
			
			# Frame rate control
			elapsed = time.time() - start_time
			sleep_time = max(0, frame_time - elapsed)
			time.sleep(sleep_time)
		
	def on_closing(self):
		"""Handle application closing"""
		self.stop_playback = True
		self.is_playing = False
		self.inference_worker.stop()
		
		# Wait for threads to finish
		if self.play_thread and self.play_thread.is_alive():
			self.play_thread.join(timeout=1.0)
			
		if self.cap:
			self.cap.release()
			
		self.root.destroy()
		
	def run(self):
		"""Start the GUI application"""
		self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
		
		# Bind resize event to update display
		self.canvas.bind('<Configure>', lambda e: self.root.after(100, self.update_canvas))
		
		# Initial display update
		self.root.after(100, lambda: self.load_video(self.video_var.get()) if self.video_var.get() != "No videos found" else None)
		
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
	
	try:
		app = SeatBinnerGUI()
		app.run()
	except Exception as e:
		print(f"Error starting application: {e}")

if __name__ == "__main__":
	main()
