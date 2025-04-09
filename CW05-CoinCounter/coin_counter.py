import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Global variables
image_directory = "CW05-CoinCounter/pliki"
images = []
current_image_index = 0

Coins = { 'onTray': {}, 'offTray': {} }
Tray = {}

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
    """Process image to detect coins and return processed image and detected circles"""
    result = image.copy()
    
    # Process grayscale differently depending on whether detecting coins on tray
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
        
        # Save the weighted sum for tray detection
        gray_processed = weighted_sum
        
    else:
        # For non-tray mode skip the custom grayscale processing
        processed = result.copy()
        gray_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    processed_float = processed.astype(np.float32) / 255.0
    processed_float = 0.5 + 2.0 * (processed_float - 0.5)  # Apply contrast level 2.0 
    processed_float = np.clip(processed_float, 0, 1)
    processed = (processed_float * 255).astype(np.uint8)
    
    # Apply blur level 5
    k_size = 11  # 5*2+1
    processed = cv2.GaussianBlur(processed, (k_size, k_size), 0)
    
    # Apply morphological closing level 5
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
    
    # Return the processed image and detected circles
    return processed, gray_processed, circles

def detect_tray(gray_image):
    """Detect tray in the processed grayscale image and return its bounding box"""
    # Apply threshold to isolate the tray, almost white
    _, thresh = cv2.threshold(gray_image, 200, 200, cv2.THRESH_BINARY)

    # Apply morphological operations for a cleaner contour 
    kernel = np.ones((17,17), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return empty dictionary
    if not contours:
        return {}
    
    # Find the largest contour (which should be the tray)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Return the tray information
    return {
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'contour': largest_contour
    }

def is_point_inside_tray(point, tray):
    """Check if point (x, y) is inside the tray contour"""
    if not tray:
        return False
    result = cv2.pointPolygonTest(tray['contour'], point, False)
    return result >= 0  # True if point is inside or on the contour

def mark_coins(image, on_tray_circles, off_tray_circles, tray):
    """Mark coins and tray on the image"""
    result = image.copy()
    
    # Draw tray contour if detected
    if tray and 'contour' in tray:
        # Draw the contour in blue
        cv2.drawContours(result, [tray['contour']], 0, (255, 0, 0), 2)
    # Fallback to bounding box if contour is not available
    elif tray and 'x' in tray:
        cv2.rectangle(result, 
                     (tray['x'], tray['y']), 
                     (tray['x'] + tray['width'], tray['y'] + tray['height']), 
                     (255, 0, 0), 2)  # Blue rectangle
    
    # Draw on-tray circles
    if on_tray_circles is not None:
        on_tray_circles = np.uint16(np.around(on_tray_circles))
        for i in on_tray_circles[0, :]:
            center = (int(i[0]), int(i[1]))
            radius = int(i[2])
            
            # Green for coins on tray
            cv2.circle(result, center, radius, (0, 255, 0), 2)  # Circle outline
            cv2.circle(result, center, 2, (0, 255, 0), 3)  # Circle center
    
    # Draw off-tray circles
    if off_tray_circles is not None:
        off_tray_circles = np.uint16(np.around(off_tray_circles))
        for i in off_tray_circles[0, :]:
            center = (int(i[0]), int(i[1]))
            radius = int(i[2])
            
            # Red for coins off tray
            cv2.circle(result, center, radius, (0, 0, 255), 2)  # Circle outline
            cv2.circle(result, center, 2, (0, 0, 255), 3)  # Circle center
    
    return result

def coin_pipeline(image):
    """Main coin detection pipeline"""
    global Coins, Tray
    
    # Clear previous data
    Coins['onTray'] = {}
    Coins['offTray'] = {}
    
    # Step 1: Detect coins on tray and get processed image for tray detection
    processed, gray_processed, on_tray_circles = detect_coins(image, onTray=True)
    
    # Step 2: Detect tray using the processed image
    Tray = detect_tray(gray_processed)
    
    # Step 3: Detect all coins (will include both on and off tray)
    _, _, all_circles = detect_coins(image, onTray=False)
    
    # Step 4: Filter to get only off-tray coins
    off_tray_circles = None
    if all_circles is not None:
        # Convert to numpy arrays
        all_circles_arr = np.uint16(np.around(all_circles))
        
        # Create a filtered list for off-tray coins
        off_tray_list = []
        
        for circle in all_circles_arr[0, :]:
            center = (int(circle[0]), int(circle[1]))
            # If center is outside tray contour, add to off_tray_list
            if not is_point_inside_tray(center, Tray):
                off_tray_list.append(circle)
        
        # Convert filtered list back to proper format if not empty
        if off_tray_list:
            off_tray_circles = np.array([off_tray_list], dtype=np.float32)
    
    # Step 5: Populate Coins dictionaries
    if on_tray_circles is not None:
        on_tray_circles_arr = np.uint16(np.around(on_tray_circles))
        for idx, circle in enumerate(on_tray_circles_arr[0, :]):
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            Coins['onTray'][idx] = {
                'center': center,
                'radius': radius
            }
    
    if off_tray_circles is not None:
        off_tray_circles_arr = np.uint16(np.around(off_tray_circles))
        for idx, circle in enumerate(off_tray_circles_arr[0, :]):
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            Coins['offTray'][idx] = {
                'center': center,
                'radius': radius
            }
    
    # Step 6: Mark up the image with coins and tray
    marked_image = mark_coins(image, on_tray_circles, off_tray_circles, Tray)
    
    return marked_image

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
        
        # Apply coin pipeline
        processed_image = coin_pipeline(image)
        
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
        
        # Update status with coin counts
        status_var.set(f"Image {current_image_index + 1} of {len(images)} | "
                      f"Coins on tray: {len(Coins['onTray'])} | "
                      f"Coins off tray: {len(Coins['offTray'])}")
    
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
