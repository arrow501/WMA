from tkinter import image_names
import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Global variables
image_directory = "CW05/pliki"
images = []
current_image_index = 0

# Processing control variables
enable_blur = False
blur_amount = 5
enable_closing = False
closing_size = 5
enable_opening = False
opening_size = 5

def load_images():
    global images
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_directory, filename))
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not read image {filename}")

def resize_frame(frame, width, height):
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

def apply_image_processing(image):
    """Apply selected image processing operations based on enabled flags and slider values."""
    result = image.copy()
    
    if enable_blur:
        # Apply Gaussian blur (must be odd number)
        k_size = max(3, blur_amount * 2 + 1)
        result = cv2.GaussianBlur(result, (k_size, k_size), 0)
        
    if enable_closing:
        # Apply morphological closing
        k_size = max(3, closing_size * 2 + 1)
        kernel = np.ones((k_size, k_size), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
    if enable_opening:
        # Apply morphological opening
        k_size = max(3, opening_size * 2 + 1)
        kernel = np.ones((k_size, k_size), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
    return result

def create_gui():
    global current_image_index
    
    # Create the main window
    root = tk.Tk()
    root.title("Coin Counter - Image Viewer")
    
    # Set window size
    window_width = 1000
    window_height = 700
    root.geometry(f"{window_width}x{window_height}")
    
    # Main content frame
    main_content = tk.Frame(root)
    main_content.pack(fill=tk.BOTH, expand=True)
    
    # Side panel for controls
    side_panel = tk.Frame(main_content, width=200, bg="#f0f0f0", relief=tk.RAISED, borderwidth=1)
    side_panel.pack(side=tk.LEFT, fill=tk.Y)
    side_panel.pack_propagate(False)  # Prevent the frame from shrinking
    
    side_panel_title = tk.Label(side_panel, text="Image Processing", font=("Arial", 12, "bold"), bg="#f0f0f0")
    side_panel_title.pack(pady=10)
    
    # Frame for the image display
    frame_display = tk.Frame(main_content, width=window_width-200, height=window_height-100)
    frame_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    frame_display.pack_propagate(False)
    
    # Canvas for displaying the image
    canvas = tk.Canvas(frame_display, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Processing controls
    def create_processing_control(title, enable_var, slider_var, slider_range=(1, 20)):
        """Create a processing control with checkbox and slider."""
        frame = tk.Frame(side_panel, bg="#f0f0f0", pady=5)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Checkbox
        cb = tk.Checkbutton(frame, text=title, variable=enable_var, bg="#f0f0f0",
                           command=lambda: show_image())
        cb.pack(anchor=tk.W)
        
        # Slider
        slider = tk.Scale(frame, from_=slider_range[0], to=slider_range[1], orient=tk.HORIZONTAL,
                         variable=slider_var, command=lambda v: show_image())
        slider.pack(fill=tk.X)
    
    # Control variables
    blur_var = tk.BooleanVar(value=enable_blur)
    blur_slider_var = tk.IntVar(value=blur_amount)
    closing_var = tk.BooleanVar(value=enable_closing)
    closing_slider_var = tk.IntVar(value=closing_size)
    opening_var = tk.BooleanVar(value=enable_opening)
    opening_slider_var = tk.IntVar(value=opening_size)
    
    # Add processing controls
    create_processing_control("Blur", blur_var, blur_slider_var)
    create_processing_control("Closing", closing_var, closing_slider_var)
    create_processing_control("Opening", opening_var, opening_slider_var)
    
    # Controls frame for navigation
    controls_frame = tk.Frame(root, height=100)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Navigation buttons
    btn_frame = tk.Frame(controls_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
    
    prev_btn = tk.Button(btn_frame, text="Previous", command=lambda: show_image(current_image_index - 1))
    prev_btn.pack(side=tk.LEFT, padx=10)
    
    next_btn = tk.Button(btn_frame, text="Next", command=lambda: show_image(current_image_index + 1))
    next_btn.pack(side=tk.RIGHT, padx=10)
    
    # Image slider
    image_slider = tk.Scale(controls_frame, from_=0, to=len(images)-1 if images else 0,
                           orient=tk.HORIZONTAL, command=lambda v: show_image(int(v)))
    image_slider.pack(fill=tk.X, padx=10)
    
    # Status label
    status_var = tk.StringVar()
    status_label = tk.Label(controls_frame, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def show_image(index=None):
        global current_image_index, enable_blur, blur_amount, enable_closing, closing_size, enable_opening, opening_size
        if not images:
            status_var.set("No images loaded")
            return
        
        # Update processing flags based on checkboxes
        enable_blur = blur_var.get()
        blur_amount = blur_slider_var.get()
        enable_closing = closing_var.get()
        closing_size = closing_slider_var.get()
        enable_opening = opening_var.get()
        opening_size = opening_slider_var.get()
        
        # Handle index bounds
        if index is not None:
            current_image_index = max(0, min(index, len(images) - 1))
        
        # Update slider position
        image_slider.set(current_image_index)
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = frame_display.winfo_width()
            canvas_height = frame_display.winfo_height()
        
        # If dimensions are still invalid, use fallback dimensions
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Get and resize the image
        image = images[current_image_index].copy()
        
        # Apply image processing
        image = apply_image_processing(image)
        
        # Resize for display
        display_image = resize_frame(image, canvas_width, canvas_height)
        
        # Convert from OpenCV BGR to RGB for tkinter
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(rgb_image)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor=tk.CENTER)
        canvas.image = img_tk  # Keep a reference to prevent garbage collection
        
        # Update status
        status_var.set(f"Image {current_image_index + 1} of {len(images)}")
    
    # Key bindings for navigation
    def key_handler(event):
        if event.keysym == "Right":
            show_image(current_image_index + 1)
        elif event.keysym == "Left":
            show_image(current_image_index - 1)
        elif event.keysym == "Home":
            show_image(0)
        elif event.keysym == "End":
            show_image(len(images) - 1)
    
    root.bind("<Key>", key_handler)
    
    # Handle window resize
    def on_resize(event):
        if images:
            show_image()
    
    canvas.bind("<Configure>", on_resize)
    
    # Initial image display
    if images:
        show_image(0)
    else:
        status_var.set("No images loaded")
    
    return root

def main():
    # Load images from directory
    load_images()
    
    if not images:
        print("No images found in directory:", image_directory)
        return
    
    # Create and start the GUI
    root = create_gui()
    root.mainloop()

if __name__ == "__main__":
    main()