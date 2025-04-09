import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


ball_video_path = r'./CW05-ProjektAimbot/movingball.mp4'
notebook_video_path = r'./CW05-ProjektAimbot/fluorescent_notebook.mp4'

# Global variables
frames = []
current_frame_index = 0
frame_sampling_rate = 30
current_video_type = 'ball'  # Default to ball video

def load(video_path, skip_frames=30): # Load every 30th frame by default
    global frames, frame_sampling_rate
    frame_sampling_rate = skip_frames
    
    # Clear existing frames
    frames = []
    
    # Load frames from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return False
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_frames == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Loaded {len(frames)} frames (every {skip_frames}th frame)")
    return True

def get_next_frame():
    """Get the next frame in the list."""
    global current_frame_index
    if not frames or current_frame_index >= len(frames) - 1:
        return None
    
    current_frame_index += 1
    return frames[current_frame_index]

def get_prev_frame():
    """Get the previous frame in the list."""
    global current_frame_index
    if not frames or current_frame_index <= 0:
        return None
    
    current_frame_index -= 1
    return frames[current_frame_index]

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

def overlay_on_bw(raw_frame, mask):
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
    
    # Combine the colored ball and grayscale background
    return cv2.add(colored_region, grayscale_region)

def draw_contours(frame, contours, color=(0, 255, 0), thickness=2):
    """Draw contours on the frame."""
    if contours:
        frame_copy = frame.copy()
        cv2.drawContours(frame_copy, contours, -1, color, thickness)
        return frame_copy
    return frame

def draw_bounding_box(frame, rect, color=(255, 0, 0), thickness=2):
    """Draw a bounding rectangle on the frame."""
    frame_copy = frame.copy()
    x, y, w, h = rect
    cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, thickness)
    return frame_copy

def draw_crosshair(frame, center, size=10, color=(0, 255, 0), thickness=2):
    """Draw a crosshair at the center point."""
    frame_copy = frame.copy()
    center_x, center_y = center
    cv2.line(frame_copy, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
    cv2.line(frame_copy, (center_x, center_y - size), (center_x, center_y + size), color, thickness)
    return frame_copy


def mask_ball(hsv_frame):
    """Create a binary mask for the red ball."""
     # Define Ranges and create color masks for the main red ball
    lower_red_midtone = np.array([0, 100, 100])
    upper_red_midtone = np.array([5, 255, 255])

    mask_midtone = cv2.inRange(hsv_frame, lower_red_midtone, upper_red_midtone)

    lower_red_highlight = np.array([170, 100, 100])
    upper_red_highlight = np.array([190, 255, 255])

    mask_highlight = cv2.inRange(hsv_frame, lower_red_highlight, upper_red_highlight)
   
    return cv2.bitwise_or(mask_midtone, mask_highlight)

def notebook_mask(hsv_frame):
    # Use the global HSV threshold values
    global hsv_lower, hsv_upper
    
    # Create mask for yellow notebook using the current threshold values
    mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
    
    # Apply morphological operations if needed
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    
    return mask

# Default HSV values for notebook detection
hsv_lower = np.array([90, 200, 100])
hsv_upper = np.array([160, 255, 255])

def process_frame(raw_frame, draw_contours_flag=False, draw_bbox_flag=True, draw_crosshair_flag=True, 
                 use_bw_background=True, blur_strength=18, video_type='ball'):
    if raw_frame is None:
        return None, {}
    
    # Create a copy of the frame to avoid modifying the original
    frame = raw_frame.copy()
    results = {}
    
    # Apply Gaussian blur to reduce noise
    # Convert blur_strength to an odd number using the formula 2n+1
    kernel_size = 2 * blur_strength + 1
    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    # Convert the frame to HSV colorspace
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a binary mask based on selected video type
    if video_type == 'notebook':
        mask = notebook_mask(hsv_frame)
    else:  # default to ball
        mask = mask_ball(hsv_frame)

    # apply morphological close operation
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    # Create grayscale background with colored ball overlay if flag is set
    if use_bw_background:
        frame = overlay_on_bw(raw_frame, mask)
    else:
        # Use the original color frame
        frame = raw_frame.copy()

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea, default=None) if contours else None
    
    if largest_contour is not None:
        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Get the center of the bounding rectangle
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Draw markers based on flags
        if draw_contours_flag:
            frame = draw_contours(frame, contours)
        
        if draw_bbox_flag:
            frame = draw_bounding_box(frame, (x, y, w, h))
        
        if draw_crosshair_flag:
            frame = draw_crosshair(frame, (center_x, center_y))
        
        # Add results to the dictionary
        results["contours"] = contours
        results["largest_contour"] = largest_contour
        results["bounding_rect"] = (x, y, w, h)
        results["center"] = (center_x, center_y)
    
    return frame, results

def create_gui():
    global current_frame_index, current_video_type
    
    # Create the main window
    root = tk.Tk()
    root.title("Object Tracking")
    
    # Set window size (720p height)
    window_width = 1280
    window_height = 720
    root.geometry(f"{window_width}x{window_height}")
    
    # Frame for the video display
    frame_display = tk.Frame(root, width=window_width, height=window_height-100)
    frame_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    frame_display.pack_propagate(False)
    
    # Canvas for displaying the video frame
    canvas = tk.Canvas(frame_display, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Frame navigation controls
    controls_frame = tk.Frame(root, height=100)
    controls_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Checkbuttons for toggling markers
    markers_frame = tk.Frame(controls_frame)
    markers_frame.pack(side=tk.TOP, fill=tk.X)
    
    # Left side - checkboxes
    checkbox_frame = tk.Frame(markers_frame)
    checkbox_frame.pack(side=tk.LEFT, fill=tk.X)
    
    show_contours = tk.BooleanVar(value=True)
    show_bbox = tk.BooleanVar(value=True)
    show_crosshair = tk.BooleanVar(value=True)
    use_bw_background = tk.BooleanVar(value=True)
    
    contours_cb = tk.Checkbutton(checkbox_frame, text="Contours", variable=show_contours,
                                command=lambda: show_frame())
    bbox_cb = tk.Checkbutton(checkbox_frame, text="Bounding Box", variable=show_bbox,
                           command=lambda: show_frame())
    crosshair_cb = tk.Checkbutton(checkbox_frame, text="Crosshair", variable=show_crosshair,
                                command=lambda: show_frame())
    bw_background_cb = tk.Checkbutton(checkbox_frame, text="BW Background", variable=use_bw_background,
                                    command=lambda: show_frame())
    
    contours_cb.pack(side=tk.LEFT, padx=10)
    bbox_cb.pack(side=tk.LEFT, padx=10)
    crosshair_cb.pack(side=tk.LEFT, padx=10)
    bw_background_cb.pack(side=tk.LEFT, padx=10)
    
    # Middle - blur slider
    blur_frame = tk.Frame(markers_frame)
    blur_frame.pack(side=tk.LEFT, fill=tk.X, padx=10, expand=True)
    
    blur_strength = tk.IntVar(value=18)  # Default value 18 (kernel size 37)
    
    tk.Label(blur_frame, text="Blur:").pack(side=tk.LEFT, padx=5)
    blur_slider = tk.Scale(blur_frame, from_=0, to=25, orient=tk.HORIZONTAL, 
                         variable=blur_strength, length=150, command=lambda _: show_frame())
    blur_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Right side - video selection dropdown
    video_frame = tk.Frame(markers_frame)
    video_frame.pack(side=tk.RIGHT, fill=tk.X, padx=10)
    
    tk.Label(video_frame, text="Video:").pack(side=tk.LEFT, padx=5)
    
    video_options = ["Ball", "Notebook"]
    selected_video = tk.StringVar(value=video_options[0])  # Default to Ball
    
    def on_video_change(*args):
        # Handle video change
        video_choice = selected_video.get()
        global current_video_type
        
        if video_choice == "Ball":
            if not load(ball_video_path, frame_sampling_rate):
                print("Failed to load ball video")
                return
            current_video_type = 'ball'
        else:  # Notebook
            if not load(notebook_video_path, frame_sampling_rate):
                print("Failed to load notebook video")
                return
            current_video_type = 'notebook'
        
        # Update frame slider max value
        frame_slider.config(to=len(frames)-1 if frames else 0)
        
        # Reset to first frame
        show_frame(0)
    
    video_dropdown = tk.OptionMenu(video_frame, selected_video, *video_options)
    video_dropdown.pack(side=tk.LEFT, padx=5)
    
    # Add HSV tuner button
    hsv_tuner_button = tk.Button(video_frame, text="Tune HSV", 
                               command=lambda: create_hsv_tuner(root))
    hsv_tuner_button.pack(side=tk.LEFT, padx=5)
    
    # Link the variable to the callback
    selected_video.trace_add("write", on_video_change)
    
    # Add handler for the refresh view event
    root.bind("<<RefreshView>>", lambda e: show_frame())
    
    # Frame slider
    frame_slider = tk.Scale(controls_frame, from_=0, to=len(frames)-1 if frames else 0,
                           orient=tk.HORIZONTAL, command=lambda v: show_frame(int(v)))
    frame_slider.pack(fill=tk.X, padx=10)
    
    # Function to display a frame
    def show_frame(index=None):
        global current_frame_index
        if index is not None:
            current_frame_index = index
        
        if not frames or current_frame_index < 0 or current_frame_index >= len(frames):
            return
        
        # Get the frame
        frame = frames[current_frame_index]
        
        # Process the frame through the image processing pipeline with marker flags and blur strength
        processed_frame, results = process_frame(
            frame, 
            show_contours.get(), 
            show_bbox.get(), 
            show_crosshair.get(),
            use_bw_background.get(),
            blur_strength.get(),
            current_video_type  # Pass the current video type
        )
        
        # Update slider position
        frame_slider.set(current_frame_index)
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = frame_display.winfo_width()
            canvas_height = frame_display.winfo_height()
        
        # If dimensions are still invalid, use fallback dimensions
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 1280
            canvas_height = 720
        
        # Resize frame to fit the canvas
        display_frame = resize_frame(processed_frame, canvas_width, canvas_height)
        
        # Convert from OpenCV BGR to RGB for tkinter
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk, anchor=tk.CENTER)
        canvas.image = img_tk  # Keep a reference to prevent garbage collection
    
    # Key bindings for navigation
    def key_handler(event):
        if event.keysym == "Right":
            new_index = min(current_frame_index + 1, len(frames) - 1)
            show_frame(new_index)
        elif event.keysym == "Left":
            new_index = max(current_frame_index - 1, 0)
            show_frame(new_index)
        elif event.keysym == "Home":
            show_frame(0)
        elif event.keysym == "End":
            show_frame(len(frames) - 1)
    
    root.bind("<Key>", key_handler)
    
    # Handle window resize
    def on_resize(event):
        if frames:
            show_frame()
    
    canvas.bind("<Configure>", on_resize)
    
    # Initial frame display
    if frames:
        show_frame(0)
    
    return root

def create_hsv_tuner(parent):
    """Create a window for tuning HSV values with live preview."""
    global hsv_lower, hsv_upper
    
    # Create a new top-level window
    tuner = tk.Toplevel(parent)
    tuner.title("HSV Tuner")
    tuner.geometry("800x600")
    
    # Split the window into two frames
    controls_frame = tk.Frame(tuner)
    controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
    
    preview_frame = tk.Frame(tuner)
    preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Canvas for displaying the preview
    preview_canvas = tk.Canvas(preview_frame, bg="black")
    preview_canvas.pack(fill=tk.BOTH, expand=True)
    
    # Create variables for HSV values
    h_min = tk.IntVar(value=hsv_lower[0])
    s_min = tk.IntVar(value=hsv_lower[1])
    v_min = tk.IntVar(value=hsv_lower[2])
    
    h_max = tk.IntVar(value=hsv_upper[0])
    s_max = tk.IntVar(value=hsv_upper[1])
    v_max = tk.IntVar(value=hsv_upper[2])
    
    # Function to update HSV values and refresh the preview
    def update_hsv(*args):
        global hsv_lower, hsv_upper
        
        # Update global HSV values
        hsv_lower = np.array([h_min.get(), s_min.get(), v_min.get()])
        hsv_upper = np.array([h_max.get(), s_max.get(), v_max.get()])
        
        # Update the preview if we have frames
        if frames and current_frame_index < len(frames):
            update_preview()
    
    # Create sliders for HSV values
    tk.Label(controls_frame, text="HSV Range Controls", font=("Arial", 14, "bold")).pack(pady=10)
    
    # Hue controls (0-179 in OpenCV)
    hue_frame = tk.LabelFrame(controls_frame, text="Hue", padx=10, pady=5)
    hue_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(hue_frame, text="Min:").pack(side=tk.LEFT)
    tk.Scale(hue_frame, from_=0, to=179, orient=tk.HORIZONTAL, 
             variable=h_min, command=lambda _: update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    tk.Label(hue_frame, text="Max:").pack(side=tk.LEFT)
    tk.Scale(hue_frame, from_=0, to=179, orient=tk.HORIZONTAL,
             variable=h_max, command=lambda _: update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Saturation controls (0-255)
    sat_frame = tk.LabelFrame(controls_frame, text="Saturation", padx=10, pady=5)
    sat_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(sat_frame, text="Min:").pack(side=tk.LEFT)
    tk.Scale(sat_frame, from_=0, to=255, orient=tk.HORIZONTAL,
             variable=s_min, command=lambda _: update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    tk.Label(sat_frame, text="Max:").pack(side=tk.LEFT)
    tk.Scale(sat_frame, from_=0, to=255, orient=tk.HORIZONTAL,
             variable=s_max, command=lambda _: update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Value controls (0-255)
    val_frame = tk.LabelFrame(controls_frame, text="Value (Brightness)", padx=10, pady=5)
    val_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(val_frame, text="Min:").pack(side=tk.LEFT)
    tk.Scale(val_frame, from_=0, to=255, orient=tk.HORIZONTAL,
             variable=v_min, command=lambda _: update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    tk.Label(val_frame, text="Max:").pack(side=tk.LEFT)
    tk.Scale(val_frame, from_=0, to=255, orient=tk.HORIZONTAL,
             variable=v_max, command=lambda _: update_hsv()).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Save and reset buttons
    buttons_frame = tk.Frame(controls_frame)
    buttons_frame.pack(fill=tk.X, pady=10)
    
    def reset_values():
        # Reset to default values
        h_min.set(90)
        s_min.set(200)
        v_min.set(100)
        h_max.set(160)
        s_max.set(255)
        v_max.set(255)
        update_hsv()
    
    # Current values display
    values_frame = tk.LabelFrame(controls_frame, text="Current Values", padx=10, pady=5)
    values_frame.pack(fill=tk.X, pady=10)
    
    values_text = tk.Text(values_frame, height=2, width=30)
    values_text.pack(fill=tk.X)
    
    def update_values_text():
        values_text.delete(1.0, tk.END)
        values_text.insert(tk.END, f"Lower: [{h_min.get()}, {s_min.get()}, {v_min.get()}]\n")
        values_text.insert(tk.END, f"Upper: [{h_max.get()}, {s_max.get()}, {v_max.get()}]")
        # Schedule next update
        tuner.after(100, update_values_text)
    
    # Start updating the text
    update_values_text()
    
    tk.Button(buttons_frame, text="Reset to Default", command=reset_values).pack(side=tk.LEFT, padx=5)
    tk.Button(buttons_frame, text="Apply to Main View", command=lambda: parent.event_generate("<<RefreshView>>")).pack(side=tk.RIGHT, padx=5)
    
    def update_preview():
        """Update the preview image with current HSV settings."""
        if not frames or current_frame_index >= len(frames):
            return
        
        frame = frames[current_frame_index]
        if frame is None:
            return
        
        # Process the frame to show the mask
        blurred = cv2.GaussianBlur(frame, (37, 37), 0)  # Using default blur
        hsv_frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Get the mask with current HSV values
        mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        
        # Apply the mask to show the detected area in color
        preview_img = overlay_on_bw(frame, mask)
        
        # Resize to fit the canvas
        preview_width = preview_canvas.winfo_width()
        preview_height = preview_canvas.winfo_height()
        
        if preview_width <= 1 or preview_height <= 1:
            preview_width = 400
            preview_height = 300
        
        display_frame = resize_frame(preview_img, preview_width, preview_height)
        
        # Convert to PhotoImage
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        preview_canvas.delete("all")
        preview_canvas.create_image(preview_width//2, preview_height//2, image=img_tk, anchor=tk.CENTER)
        preview_canvas.image = img_tk  # Keep a reference
    
    # Handle window resize for preview
    def on_preview_resize(event):
        update_preview()
    
    preview_canvas.bind("<Configure>", on_preview_resize)
    
    # Initial preview
    tuner.after(100, update_preview)
    
    return tuner

def main():
    # Load frames from the default video (ball)
    if not load(ball_video_path, frame_sampling_rate):
        print("Failed to load video frames")
        return
    
    # Create and start the GUI
    root = create_gui()
    root.mainloop()

if __name__ == "__main__":
    main()