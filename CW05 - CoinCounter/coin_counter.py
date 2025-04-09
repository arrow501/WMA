import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Global variables
image_directory = "CW05/pliki"
images = []
current_image_index = 0

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

def detect_coins(image, onTray=True):
    """Apply the coin detection preset and return image with coin markings"""
    result = image.copy()
    
    # Process grayscale differently depending on whether coins are on tray
    if onTray:
        # Convert to grayscale with custom weights
        b, g, r = cv2.split(result)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)
        
        # Apply weights from the preset
        weighted_sum = r * 1.0 + g * -1.0 + b * -0.9
        
        # Apply gain
        weighted_sum = weighted_sum * 3.0
        
        # Normalize to valid range [0, 255]
        weighted_sum = np.clip(weighted_sum, 0, 255).astype(np.uint8)
        
        # Convert to 3-channel grayscale
        processed = cv2.cvtColor(weighted_sum, cv2.COLOR_GRAY2BGR)
        
    else:
        # For non-tray mode, still apply contrast enhancement to match original behavior
        processed = result.copy()

    processed_float = processed.astype(np.float32) / 255.0
    processed_float = 0.5 + 2.0 * (processed_float - 0.5)  # Apply contrast level 2.0 as in preset
    processed_float = np.clip(processed_float, 0, 1)
    processed = (processed_float * 255).astype(np.uint8)
    
    # Apply blur
    k_size = 11  # 5*2+1
    processed = cv2.GaussianBlur(processed, (k_size, k_size), 0)
    
    # Apply morphological closing
    k_size = 11  # 5*2+1
    kernel = np.ones((k_size, k_size), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Apply Hough Circle Transform with preset parameters
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, 
        dp=1,
        minDist=36,  # from preset
        param1=51,   # from preset
        param2=38,   # from preset
        minRadius=10,
        maxRadius=0
    )
    
    # Draw circles on the original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw circle outline
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw circle center
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    return result

def create_gui():
    global current_image_index
    
    # Create the main window
    root = tk.Tk()
    root.title("Coin Counter")
    
    # Set window size
    window_width = 1000
    window_height = 700
    root.geometry(f"{window_width}x{window_height}")
    
    # Main content frame
    main_content = tk.Frame(root)
    main_content.pack(fill=tk.BOTH, expand=True)
    
    # Frame for the image display
    frame_display = tk.Frame(main_content)
    frame_display.pack(fill=tk.BOTH, expand=True)
    
    # Canvas for displaying the image
    canvas = tk.Canvas(frame_display, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Controls frame for navigation
    controls_frame = tk.Frame(root, height=50)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Navigation buttons
    btn_frame = tk.Frame(controls_frame)
    btn_frame.pack(fill=tk.X, pady=10)
    
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
        global current_image_index
        
        if not images:
            status_var.set("No images loaded")
            return
        
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
        
        # Get the image
        image = images[current_image_index].copy()
        
        # Apply coin detection
        processed_image = detect_coins(image)
        
        # Resize for display
        display_image = resize_frame(processed_image, canvas_width, canvas_height)
        
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
