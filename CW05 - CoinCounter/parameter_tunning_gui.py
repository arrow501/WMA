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
enable_circle_detection = False
min_circle_dist = 50
circle_param1 = 128
circle_param2 = 64

# Grayscale conversion variables
enable_grayscale = False
grayscale_gain = 1.0  # New variable for grayscale gain
r_weight = 0.299
g_weight = 0.587
b_weight = 0.114

# New image enhancement variables
enable_contrast = False
contrast_level = 1.0
enable_sharpen = False
sharpen_amount = 1.0

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

def apply_contrast(image, amount):
    """Apply contrast adjustment to image"""
    result = image.copy()
    # Convert to float for calculations
    result_float = result.astype(np.float32) / 255.0
    # Apply contrast: f(x) = 127 + amount * (x - 127)
    result_float = 0.5 + amount * (result_float - 0.5)
    # Clip to [0, 1] range
    result_float = np.clip(result_float, 0, 1)
    # Convert back to uint8
    result = (result_float * 255).astype(np.uint8)
    return result

def apply_sharpen(image, amount):
    """Apply sharpening to image"""
    # Create a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Scale kernel based on amount
    if amount != 1.0:
        kernel = np.ones((3, 3), np.float32) * (-amount/8)
        kernel[1, 1] = amount + 1
        
    # Apply the kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def apply_image_processing(image):
    """Apply selected image processing operations based on enabled flags and slider values."""
    result = image.copy()
    
    # Apply grayscale conversion with custom weights if enabled
    if enable_grayscale:
        # Split the image into BGR channels
        b, g, r = cv2.split(result)
        
        # Convert channels to float32 for calculations with negative weights
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)
        
        # Apply custom weights including negative values
        weighted_sum = b * b_weight + g * g_weight + r * r_weight
        
        # Apply gain
        weighted_sum = weighted_sum * grayscale_gain
        
        # Normalize to valid range [0, 255]
        weighted_sum = np.clip(weighted_sum, 0, 255).astype(np.uint8)
        
        # Convert to 3-channel grayscale (for consistent processing)
        result = cv2.cvtColor(weighted_sum, cv2.COLOR_GRAY2BGR)
    
    # Apply contrast adjustment if enabled (enhances edges for circle detection)
    if enable_contrast:
        result = apply_contrast(result, contrast_level)
    
    # Apply sharpening if enabled (enhances edges for circle detection)
    if enable_sharpen:
        result = apply_sharpen(result, sharpen_amount)
        
    # Apply blur (reduces noise which helps circle detection)
    if enable_blur:
        # Apply Gaussian blur (must be odd number)
        k_size = max(3, blur_amount * 2 + 1)
        result = cv2.GaussianBlur(result, (k_size, k_size), 0)
    
    # Apply morphological operations
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
    
    # Detect circles if enabled
    circles = None
    if enable_circle_detection:
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            dp=1,
            minDist=min_circle_dist,
            param1=circle_param1,
            param2=circle_param2,
            minRadius=10,
            maxRadius=0
        )
        
        # Draw circles on the image if any were found
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
    root.title("Coin Counter - Processing Tuning")
    
    # Set window size
    window_width = 1200
    window_height = 800
    root.geometry(f"{window_width}x{window_height}")
    
    # Define all control variables
    grayscale_var = tk.BooleanVar(value=enable_grayscale)
    grayscale_gain_var = tk.DoubleVar(value=grayscale_gain)
    r_weight_var = tk.DoubleVar(value=r_weight)
    g_weight_var = tk.DoubleVar(value=g_weight)
    b_weight_var = tk.DoubleVar(value=b_weight)
    
    contrast_var = tk.BooleanVar(value=enable_contrast)
    contrast_slider_var = tk.DoubleVar(value=contrast_level)
    sharpen_var = tk.BooleanVar(value=enable_sharpen)
    sharpen_slider_var = tk.DoubleVar(value=sharpen_amount)
    
    blur_var = tk.BooleanVar(value=enable_blur)
    blur_slider_var = tk.IntVar(value=blur_amount)
    
    closing_var = tk.BooleanVar(value=enable_closing)
    closing_slider_var = tk.IntVar(value=closing_size)
    opening_var = tk.BooleanVar(value=enable_opening)
    opening_slider_var = tk.IntVar(value=opening_size)
    
    circle_detection_var = tk.BooleanVar(value=enable_circle_detection)
    min_dist_var = tk.IntVar(value=min_circle_dist)
    param1_var = tk.IntVar(value=circle_param1)
    param2_var = tk.IntVar(value=circle_param2)
    
    # Main content frame
    main_content = tk.Frame(root)
    main_content.pack(fill=tk.BOTH, expand=True)
    
    # Side panel for controls - make it wider but more compact
    sidebar_width = 400
    side_panel = tk.Frame(main_content, width=sidebar_width, bg="#f0f0f0", relief=tk.RAISED, borderwidth=1)
    side_panel.pack(side=tk.LEFT, fill=tk.Y)
    side_panel.pack_propagate(False)  # Prevent the frame from shrinking
    
    # Add a scrollable canvas for the sidebar
    sidebar_canvas = tk.Canvas(side_panel, bg="#f0f0f0", highlightthickness=0)
    sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Scrollbar for canvas
    scrollbar = tk.Scrollbar(side_panel, orient=tk.VERTICAL, command=sidebar_canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    sidebar_canvas.configure(yscrollcommand=scrollbar.set)
    
    # Frame inside canvas for controls
    controls_container = tk.Frame(sidebar_canvas, bg="#f0f0f0")
    controls_container.bind("<Configure>", lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
    sidebar_canvas.create_window((0, 0), window=controls_container, anchor=tk.NW, width=sidebar_width-20)
    
    # Enable mouse wheel scrolling
    def _on_mousewheel(event):
        sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    side_panel_title = tk.Label(controls_container, text="Image Processing", font=("Arial", 12, "bold"), bg="#f0f0f0")
    side_panel_title.pack(pady=5)
    
    # Frame for the image display - adjust width for wider sidebar
    frame_display = tk.Frame(main_content, width=window_width-sidebar_width, height=window_height-100)
    frame_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    frame_display.pack_propagate(False)
    
    # Canvas for displaying the image
    canvas = tk.Canvas(frame_display, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Add grayscale gain variable
    grayscale_gain_var = tk.DoubleVar(value=grayscale_gain)
    
    # Create a compact control row (checkbox, title, slider all in one row)
    def create_compact_control(parent, title, enable_var, slider_var, slider_range=(1, 20), resolution=1):
        """Create an ultra-compact single row control with checkbox, title, and slider"""
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=2, pady=1)
        
        # Checkbox
        cb = tk.Checkbutton(frame, variable=enable_var, bg="#f0f0f0", 
                           command=lambda: show_image(), padx=0)
        cb.pack(side=tk.LEFT)
        
        # Title label (smaller font)
        label = tk.Label(frame, text=title, bg="#f0f0f0", font=("Arial", 8), width=11, anchor=tk.W)
        label.pack(side=tk.LEFT, padx=(0, 2))
        
        # Value display
        value_label = tk.Label(frame, textvariable=slider_var, bg="#f0f0f0", 
                             width=4, font=("Arial", 7))
        value_label.pack(side=tk.RIGHT, padx=(0, 2))
        
        # Slider
        slider = tk.Scale(frame, from_=slider_range[0], to=slider_range[1], resolution=resolution,
                        orient=tk.HORIZONTAL, variable=slider_var, command=lambda v: show_image(),
                        showvalue=0, width=8)  # Thin slider with no value display
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        return frame
    
    # For parameters without enable/disable, even more compact
    def create_param_row(parent, title, slider_var, slider_range=(1, 20), resolution=1):
        """Create an ultra-compact parameter row with just title and slider"""
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=(15, 2), pady=1)
        
        # Title label (smaller font, indented)
        label = tk.Label(frame, text=title, bg="#f0f0f0", font=("Arial", 8), width=9, anchor=tk.W)
        label.pack(side=tk.LEFT)
        
        # Value display
        value_label = tk.Label(frame, textvariable=slider_var, bg="#f0f0f0", 
                             width=4, font=("Arial", 7))
        value_label.pack(side=tk.RIGHT, padx=(0, 2))
        
        # Slider
        slider = tk.Scale(frame, from_=slider_range[0], to=slider_range[1], resolution=resolution,
                        orient=tk.HORIZONTAL, variable=slider_var, command=lambda v: show_image(),
                        showvalue=0, width=8)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        return frame
    
    # Section header - very compact
    def add_section_header(parent, text):
        header = tk.Label(parent, text=text, font=("Arial", 8, "bold"), 
                        bg="#e0e0e0", fg="#333")
        header.pack(fill=tk.X, pady=(3,1), padx=0)
        
    # Redesigned, ultra-compact layout
    main_panel = tk.Frame(controls_container, bg="#f0f0f0")
    main_panel.pack(fill=tk.BOTH, expand=True)
    
    # 1. Grayscale Section
    add_section_header(main_panel, "1. Grayscale Conversion")
    grayscale_frame = tk.Frame(main_panel, bg="#f0f0f0")
    grayscale_frame.pack(fill=tk.X)
    
    # Grayscale with its own gain slider
    create_compact_control(grayscale_frame, "Grayscale", grayscale_var, grayscale_gain_var, 
                         slider_range=(0.1, 3.0), resolution=0.1)
    
    # RGB weight controls in rows
    rgb_frame = tk.Frame(grayscale_frame, bg="#f0f0f0")
    rgb_frame.pack(fill=tk.X)
    create_param_row(rgb_frame, "Red", r_weight_var, slider_range=(-1.0, 1.0), resolution=0.1)
    create_param_row(rgb_frame, "Green", g_weight_var, slider_range=(-1.0, 1.0), resolution=0.1)
    create_param_row(rgb_frame, "Blue", b_weight_var, slider_range=(-1.0, 1.0), resolution=0.1)
    
    # 2. Image Enhancement
    add_section_header(main_panel, "2. Image Enhancement")
    enhance_frame = tk.Frame(main_panel, bg="#f0f0f0")
    enhance_frame.pack(fill=tk.X)
    
    create_compact_control(enhance_frame, "Contrast", contrast_var, contrast_slider_var, 
                         slider_range=(0.5, 2.0), resolution=0.1)
    create_compact_control(enhance_frame, "Sharpen", sharpen_var, sharpen_slider_var,
                         slider_range=(0.1, 5.0), resolution=0.1)
    
    # 3. Noise Reduction
    add_section_header(main_panel, "3. Noise Reduction")
    noise_frame = tk.Frame(main_panel, bg="#f0f0f0")
    noise_frame.pack(fill=tk.X)
    
    create_compact_control(noise_frame, "Blur", blur_var, blur_slider_var, 
                         slider_range=(1, 20))
    
    # 4. Morphological Operations
    add_section_header(main_panel, "4. Morphological Operations")
    morph_frame = tk.Frame(main_panel, bg="#f0f0f0")
    morph_frame.pack(fill=tk.X)
    
    create_compact_control(morph_frame, "Closing", closing_var, closing_slider_var, 
                         slider_range=(1, 20))
    create_compact_control(morph_frame, "Opening", opening_var, opening_slider_var, 
                         slider_range=(1, 20))
    
    # 5. Circle Detection
    add_section_header(main_panel, "5. Circle Detection")
    circle_frame = tk.Frame(main_panel, bg="#f0f0f0")
    circle_frame.pack(fill=tk.X)
    
    create_compact_control(circle_frame, "Detect", circle_detection_var, min_dist_var, 
                         slider_range=(10, 100))
    
    # Circle detection parameters
    create_param_row(circle_frame, "Edge Param", param1_var, slider_range=(10, 300))
    create_param_row(circle_frame, "Circle Param", param2_var, slider_range=(1, 100))
    
    # Controls frame for navigation
    controls_frame = tk.Frame(root, height=100)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Navigation buttons
    btn_frame = tk.Frame(controls_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
    
    # Navigation buttons (left side)
    nav_left_frame = tk.Frame(btn_frame)
    nav_left_frame.pack(side=tk.LEFT, padx=10)
    
    prev_btn = tk.Button(nav_left_frame, text="Previous", command=lambda: show_image(current_image_index - 1))
    prev_btn.pack(side=tk.LEFT)
    
    # Preset buttons (center)
    presets_frame = tk.Frame(btn_frame)
    presets_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
    
    # Function to create preset buttons
    def create_preset_button(title, settings_dict):
        def apply_preset():
            # Apply all settings from the dictionary - use variable names directly
            for setter_func, value in settings_dict:
                setter_func(value)
            show_image()
        
        # Create the button directly in presets_frame and pack it side-by-side (LEFT)
        btn = tk.Button(presets_frame, text=title, command=apply_preset,
                       bg="#e0e0ff", font=("Arial", 9, "bold"))
        btn.pack(side=tk.LEFT, padx=5, pady=2)
        return btn
    
    # Coin detection preset
    coin_preset_settings = [
        (grayscale_var.set, True),
        (grayscale_gain_var.set, 3.0),
        (r_weight_var.set, 1.0),
        (g_weight_var.set, -1.0),
        (b_weight_var.set, -0.9),
        (contrast_var.set, True),
        (contrast_slider_var.set, 2.0),
        (blur_var.set, True),
        (blur_slider_var.set, 5),
        (closing_var.set, True),
        (closing_slider_var.set, 5),
        (opening_var.set, False),
        (circle_detection_var.set, True),
        (min_dist_var.set, 36),
        (param1_var.set, 51),
        (param2_var.set, 38)
    ]
    
    coin_preset_btn = create_preset_button("Coins on Tray", coin_preset_settings)
    
    # Add a new preset for coins off tray - identical but without grayscale
    coins_off_tray_settings = [
        (grayscale_var.set, False),  # The only difference: grayscale disabled
        (grayscale_gain_var.set, 3.0),
        (r_weight_var.set, 1.0),
        (g_weight_var.set, -1.0),
        (b_weight_var.set, -0.9),
        (contrast_var.set, True),
        (contrast_slider_var.set, 2.0),
        (blur_var.set, True),
        (blur_slider_var.set, 5),
        (closing_var.set, True),
        (closing_slider_var.set, 5),
        (opening_var.set, False),
        (circle_detection_var.set, True),
        (min_dist_var.set, 36),
        (param1_var.set, 51),
        (param2_var.set, 38)
    ]
    
    coins_off_tray_btn = create_preset_button("Coins Off Tray", coins_off_tray_settings)
    
    # Navigation buttons (right side)
    nav_right_frame = tk.Frame(btn_frame)
    nav_right_frame.pack(side=tk.RIGHT, padx=10)
    
    next_btn = tk.Button(nav_right_frame, text="Next", command=lambda: show_image(current_image_index + 1))
    next_btn.pack(side=tk.RIGHT)
    
    # Image slider
    image_slider = tk.Scale(controls_frame, from_=0, to=len(images)-1 if images else 0,
                           orient=tk.HORIZONTAL, command=lambda v: show_image(int(v)))
    image_slider.pack(fill=tk.X, padx=10)
    
    # Status label
    status_var = tk.StringVar()
    status_label = tk.Label(controls_frame, textvariable=status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def show_image(index=None):
        global current_image_index, enable_blur, blur_amount, enable_closing, closing_size
        global enable_opening, opening_size, enable_circle_detection, min_circle_dist
        global circle_param1, circle_param2, enable_grayscale, r_weight, g_weight, b_weight
        global enable_contrast, contrast_level, enable_sharpen, sharpen_amount, grayscale_gain
        
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
        enable_circle_detection = circle_detection_var.get()
        min_circle_dist = min_dist_var.get()
        circle_param1 = param1_var.get()
        circle_param2 = param2_var.get()
        enable_grayscale = grayscale_var.get()
        r_weight = r_weight_var.get()
        g_weight = g_weight_var.get()
        b_weight = b_weight_var.get()
        enable_contrast = contrast_var.get()
        contrast_level = contrast_slider_var.get()
        sharpen_amount = sharpen_slider_var.get()
        enable_sharpen = sharpen_var.get()
        grayscale_gain = grayscale_gain_var.get()
        
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