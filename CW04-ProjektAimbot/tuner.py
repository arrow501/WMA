import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

class HSVTuner:
    def __init__(self, root=None):
        """Initialize the HSV tuner with optional parent window."""
        self.frames = []
        self.current_frame_index = 0
        
        # Default HSV values
        self.hsv_lower = np.array([90, 200, 100])
        self.hsv_upper = np.array([160, 255, 255])
        
        # Create main window if no parent is provided
        if root is None:
            self.root = tk.Tk()
            self.root.title("HSV Tuner")
            self.root.geometry("1000x700")
            self.is_standalone = True
        else:
            self.root = tk.Toplevel(root)
            self.root.title("HSV Tuner")
            self.root.geometry("800x600")
            self.is_standalone = False
        
        self.create_interface()
        
        # Load default test image if in standalone mode
        if self.is_standalone:
            self.load_default_image()
    
    def create_interface(self):
        """Create the tuner interface."""
        # Split the window into two frames
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.preview_frame = tk.Frame(self.root)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for displaying the preview
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create variables for HSV values
        self.h_min = tk.IntVar(value=self.hsv_lower[0])
        self.s_min = tk.IntVar(value=self.hsv_lower[1])
        self.v_min = tk.IntVar(value=self.hsv_lower[2])
        
        self.h_max = tk.IntVar(value=self.hsv_upper[0])
        self.s_max = tk.IntVar(value=self.hsv_upper[1])
        self.v_max = tk.IntVar(value=self.hsv_upper[2])
        
        # Create sliders for HSV values
        tk.Label(self.controls_frame, text="HSV Range Controls", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File open button
        file_frame = tk.Frame(self.controls_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(file_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=5)
        
        # Hue controls (0-179 in OpenCV)
        hue_frame = tk.LabelFrame(self.controls_frame, text="Hue", padx=10, pady=5)
        hue_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(hue_frame, text="Min:").pack(side=tk.LEFT)
        tk.Scale(hue_frame, from_=0, to=179, orient=tk.HORIZONTAL, 
                variable=self.h_min, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(hue_frame, text="Max:").pack(side=tk.LEFT)
        tk.Scale(hue_frame, from_=0, to=179, orient=tk.HORIZONTAL,
                variable=self.h_max, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Saturation controls (0-255)
        sat_frame = tk.LabelFrame(self.controls_frame, text="Saturation", padx=10, pady=5)
        sat_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(sat_frame, text="Min:").pack(side=tk.LEFT)
        tk.Scale(sat_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                variable=self.s_min, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(sat_frame, text="Max:").pack(side=tk.LEFT)
        tk.Scale(sat_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                variable=self.s_max, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Value controls (0-255)
        val_frame = tk.LabelFrame(self.controls_frame, text="Value (Brightness)", padx=10, pady=5)
        val_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(val_frame, text="Min:").pack(side=tk.LEFT)
        tk.Scale(val_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                variable=self.v_min, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(val_frame, text="Max:").pack(side=tk.LEFT)
        tk.Scale(val_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                variable=self.v_max, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Current values display
        values_frame = tk.LabelFrame(self.controls_frame, text="Current Values", padx=10, pady=5)
        values_frame.pack(fill=tk.X, pady=10)
        
        self.values_text = tk.Text(values_frame, height=2, width=30)
        self.values_text.pack(fill=tk.X)
        
        # Save and reset buttons
        buttons_frame = tk.Frame(self.controls_frame)
        buttons_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(buttons_frame, text="Reset to Default", command=self.reset_values).pack(side=tk.LEFT, padx=5)
        
        if not self.is_standalone:
            tk.Button(buttons_frame, text="Apply to Main View", 
                    command=lambda: self.root.master.event_generate("<<RefreshView>>")).pack(side=tk.RIGHT, padx=5)
            
        # Morphology controls
        morph_frame = tk.LabelFrame(self.controls_frame, text="Morphology", padx=10, pady=5)
        morph_frame.pack(fill=tk.X, pady=5)
        
        self.morph_kernel_size = tk.IntVar(value=25)
        tk.Label(morph_frame, text="Kernel Size:").pack(side=tk.LEFT)
        tk.Scale(morph_frame, from_=1, to=50, orient=tk.HORIZONTAL,
                variable=self.morph_kernel_size, command=lambda _: self.update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Video navigation (only shown when video is loaded)
        self.video_controls_frame = tk.Frame(self.controls_frame)
        
        tk.Label(self.video_controls_frame, text="Video Navigation").pack(pady=5)
        
        nav_buttons = tk.Frame(self.video_controls_frame)
        nav_buttons.pack(fill=tk.X)
        
        tk.Button(nav_buttons, text="◀◀", command=lambda: self.show_frame(0)).pack(side=tk.LEFT)
        tk.Button(nav_buttons, text="◀", command=self.prev_frame).pack(side=tk.LEFT)
        tk.Button(nav_buttons, text="▶", command=self.next_frame).pack(side=tk.LEFT)
        tk.Button(nav_buttons, text="▶▶", command=lambda: self.show_frame(len(self.frames)-1)).pack(side=tk.LEFT)
        
        self.frame_slider = tk.Scale(self.video_controls_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                   command=lambda v: self.show_frame(int(v)))
        self.frame_slider.pack(fill=tk.X, pady=5)
        
        # Handle window resize for preview
        self.preview_canvas.bind("<Configure>", self.on_preview_resize)
        
        # Start updating the values text
        self.update_values_text()
    
    def open_image(self):
        """Open an image file using a file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            # Clear any existing frames
            self.frames = []
            
            # Load the selected image
            img = cv2.imread(file_path)
            if img is not None:
                self.frames = [img]
                self.current_frame_index = 0
                self.show_frame()
                
                # Hide video controls
                self.video_controls_frame.pack_forget()
            else:
                print(f"Failed to load image: {file_path}")
    
    def open_video(self):
        """Open a video file using a file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path, skip_frames=30):
        """Load frames from a video file."""
        # Clear existing frames
        self.frames = []
        
        # Load frames from the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                self.frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        print(f"Loaded {len(self.frames)} frames (every {skip_frames}th frame)")
        
        # Reset to first frame
        self.current_frame_index = 0
        
        # Show video controls and update slider
        self.video_controls_frame.pack(fill=tk.X, pady=10)
        self.frame_slider.config(to=len(self.frames)-1 if self.frames else 0)
        
        # Show the first frame
        self.show_frame()
        return True
    
    def load_default_image(self):
        """Load a default test image if available."""
        # Try a few possible locations for test images
        possible_paths = [
            './ProjektAimbot/test_image.jpg',
            './test_image.jpg',
            './ProjektAimbot/fluorescent_notebook.mp4'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if path.endswith(('.mp4', '.avi', '.mov')):
                    self.load_video(path)
                else:
                    img = cv2.imread(path)
                    if img is not None:
                        self.frames = [img]
                        self.show_frame()
                return
        
        print("No default test image found.")
    
    def next_frame(self):
        """Show the next frame in the video."""
        if self.frames and self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.show_frame()
    
    def prev_frame(self):
        """Show the previous frame in the video."""
        if self.frames and self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.show_frame()
    
    def show_frame(self, index=None):
        """Display a specific frame."""
        if index is not None:
            self.current_frame_index = index
        
        if not self.frames or self.current_frame_index >= len(self.frames):
            return
        
        # Update slider position for videos
        if len(self.frames) > 1:
            self.frame_slider.set(self.current_frame_index)
        
        # Update the preview
        self.update_preview()
    
    def update_hsv(self):
        """Update HSV values and refresh the preview."""
        # Update HSV values
        self.hsv_lower = np.array([self.h_min.get(), self.s_min.get(), self.v_min.get()])
        self.hsv_upper = np.array([self.h_max.get(), self.s_max.get(), self.v_max.get()])
        
        # Update the preview
        self.update_preview()
    
    def reset_values(self):
        """Reset HSV values to defaults."""
        self.h_min.set(90)
        self.s_min.set(200)
        self.v_min.set(100)
        self.h_max.set(160)
        self.s_max.set(255)
        self.v_max.set(255)
        self.update_hsv()
    
    def update_values_text(self):
        """Update the text display of current HSV values."""
        if hasattr(self, 'values_text'):
            self.values_text.delete(1.0, tk.END)
            self.values_text.insert(tk.END, f"Lower: [{self.h_min.get()}, {self.s_min.get()}, {self.v_min.get()}]\n")
            self.values_text.insert(tk.END, f"Upper: [{self.h_max.get()}, {self.s_max.get()}, {self.v_max.get()}]")
            
            # Schedule next update if window still exists
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(100, self.update_values_text)
    
    def overlay_on_bw(self, raw_frame, mask):
        """Overlay the colored object on a black and white background."""
        # Create a black and white background    
        bw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        # Convert the bw_frame back to BGR format (3-channel grayscale)
        bw_frame = cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2BGR)
        
        # Get the colored region from the original frame using the mask
        colored_region = cv2.bitwise_and(raw_frame, raw_frame, mask=mask)
        
        # Get the inverse of the mask for the background
        inv_mask = cv2.bitwise_not(mask)
        
        # Get the background region using the inverse mask
        grayscale_region = cv2.bitwise_and(bw_frame, bw_frame, mask=inv_mask)
        
        # Combine the colored area and grayscale background
        return cv2.add(colored_region, grayscale_region)
    
    def resize_frame(self, frame, width, height):
        """Resize the frame to the specified dimensions."""
        if frame is None:
            return None
        
        # Ensure dimensions are valid
        if width <= 0 or height <= 0:
            return frame  # Return original frame if dimensions are invalid
            
        # Calculate scaling to maintain aspect ratio
        h, w = frame.shape[:2]
        aspect = w / h
        
        if height * aspect <= width:
            # Height is the limiting factor
            new_h = height
            new_w = int(height * aspect)
        else:
            # Width is the limiting factor
            new_w = width
            new_h = int(width / aspect)
        
        # Ensure dimensions are at least 1 pixel
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    def update_preview(self):
        """Update the preview image with current HSV settings."""
        if not self.frames or self.current_frame_index >= len(self.frames):
            return
        
        frame = self.frames[self.current_frame_index].copy()
        if frame is None:
            return
        
        # Process the frame to show the mask
        blurred = cv2.GaussianBlur(frame, (37, 37), 0)  # Using default blur
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Get the mask with current HSV values
        mask = cv2.inRange(hsv_frame, self.hsv_lower, self.hsv_upper)
        
        # Apply morphological operations
        kernel_size = max(1, self.morph_kernel_size.get())
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Create a multi-panel view
        h, w = frame.shape[:2]
        
        # Original image
        preview_img = frame.copy()
        
        # Mask image (convert to BGR for consistent display)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Overlay image
        overlay = self.overlay_on_bw(frame, mask)
        
        # Stack images horizontally and vertically
        top_row = np.hstack((preview_img, mask_colored))
        bottom_row = np.hstack((overlay, np.zeros_like(overlay)))  # Empty space for symmetry
        
        # Combine into final display
        combined = np.vstack((top_row, bottom_row))
        
        # Resize to fit the canvas
        preview_width = self.preview_canvas.winfo_width()
        preview_height = self.preview_canvas.winfo_height()
        
        if preview_width <= 1 or preview_height <= 1:
            preview_width = 400
            preview_height = 300
        
        display_frame = self.resize_frame(combined, preview_width, preview_height)
        
        # Convert to PhotoImage
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(preview_width//2, preview_height//2, image=img_tk, anchor=tk.CENTER)
        self.preview_canvas.image = img_tk  # Keep a reference
    
    def on_preview_resize(self, event):
        """Handle window resize events."""
        self.update_preview()
    
    def get_hsv_values(self):
        """Return the current HSV values."""
        return self.hsv_lower, self.hsv_upper
    
    def run(self):
        """Run the main loop (for standalone mode)."""
        if self.is_standalone:
            self.root.mainloop()

def main():
    """Main function to run the HSV tuner as a standalone application."""
    tuner = HSVTuner()
    tuner.run()

if __name__ == "__main__":
    main()