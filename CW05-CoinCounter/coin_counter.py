import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Globals
IMAGE_DIR_PATH = "CW05-CoinCounter/pliki"
COIN_SIZE_THRESHOLD = 1.8  # Percentage of tray area

class CoinDetector:
    """Handles coin and tray detection in images"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset detection results"""
        self.coins_on_tray = []  # [(center, radius, is_large)]
        self.coins_off_tray = []  # [(center, radius, is_large)]
        self.tray = None
        self.tray_area = 0
    
    def process(self, image):
        """Process image to detect coins and tray"""
        # Reset results
        self.reset()
        result = image.copy()
        
        # FIRST PIPELINE: Detect on-tray coins and tray
        # 1. Apply weighted grayscale conversion for tray/on-tray coins
        b, g, r = cv2.split(image)
        weighted_gray = (r.astype(np.float32) * 1.0 + 
                         g.astype(np.float32) * -1.0 + 
                         b.astype(np.float32) * -0.9) * 3.0
        weighted_gray = np.clip(weighted_gray, 0, 255).astype(np.uint8)
        
        # 2. Process weighted image for on-tray coin detection
        processed_tray = self._enhance_image(cv2.cvtColor(weighted_gray, cv2.COLOR_GRAY2BGR))
        gray_tray = cv2.cvtColor(processed_tray, cv2.COLOR_BGR2GRAY)
        
        # 3. Detect tray using the weighted image
        self.tray = self._detect_tray(weighted_gray)
        
        # 4. Calculate tray area if tray was found
        if self.tray is not None:
            self.tray_area = cv2.contourArea(self.tray)
        
        # 5. Detect on-tray coins
        on_tray_circles = cv2.HoughCircles(
            gray_tray, cv2.HOUGH_GRADIENT, dp=1, minDist=36,
            param1=51, param2=38, minRadius=10, maxRadius=0
        )
        
        # SECOND PIPELINE: Detect all coins, filter to get off-tray coins
        # 6. Process original image without weighted grayscale
        processed_all = self._enhance_image(image)
        gray_all = cv2.cvtColor(processed_all, cv2.COLOR_BGR2GRAY)
        
        # 7. Detect all coins
        all_circles = cv2.HoughCircles(
            gray_all, cv2.HOUGH_GRADIENT, dp=1, minDist=36,
            param1=51, param2=38, minRadius=10, maxRadius=0
        )
        
        # 8. Mark on-tray coins
        if on_tray_circles is not None:
            on_tray_circles = np.uint16(np.around(on_tray_circles))
            for circle in on_tray_circles[0, :]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                
                # Classify coin size
                is_large = self._is_large_coin(radius)
                self.coins_on_tray.append((center, radius, is_large))
                
                # Color based on coin size
                if is_large:
                    # Green for large coins on tray (5zł)
                    cv2.circle(result, center, radius, (0, 255, 0), 2)
                    cv2.circle(result, center, 2, (0, 255, 0), 3)
                else:
                    # Light blue for small coins on tray (5gr)
                    cv2.circle(result, center, radius, (255, 191, 0), 2)
                    cv2.circle(result, center, 2, (255, 191, 0), 3)
        
        # 9. Filter and mark off-tray coins
        if all_circles is not None and self.tray is not None:
            all_circles = np.uint16(np.around(all_circles))
            for circle in all_circles[0, :]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                
                # Only include coins not on tray
                if not self._is_on_tray(center):
                    # Classify coin size
                    is_large = self._is_large_coin(radius)
                    self.coins_off_tray.append((center, radius, is_large))
                    
                    # Color based on coin size
                    if is_large:
                        # Red for large coins off tray (5zł)
                        cv2.circle(result, center, radius, (0, 0, 255), 2)
                        cv2.circle(result, center, 2, (0, 0, 255), 3)
                    else:
                        # Yellow for small coins off tray (5gr)
                        cv2.circle(result, center, radius, (0, 255, 255), 2)
                        cv2.circle(result, center, 2, (0, 255, 255), 3)
        
        # 10. Draw tray contour
        if self.tray is not None:
            cv2.drawContours(result, [self.tray], 0, (255, 0, 0), 2)  # Blue for tray
        
        return result
    
    def _enhance_image(self, image):
        """Enhance image for better detection"""
        # Enhance contrast
        img_float = image.astype(np.float32) / 255.0
        img_float = 0.5 + 2.0 * (img_float - 0.5)
        img_float = np.clip(img_float, 0, 1)
        enhanced = (img_float * 255).astype(np.uint8)
        
        # Apply blur and morphology
        enhanced = cv2.GaussianBlur(enhanced, (11, 11), 0)
        kernel = np.ones((11, 11), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def _detect_tray(self, gray):
        """Detect tray in grayscale image"""
        # Threshold to isolate the tray
        _, thresh = cv2.threshold(gray, 200, 200, cv2.THRESH_BINARY)
        
        # Clean up with morphology
        kernel = np.ones((17, 17), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
    
    def _is_on_tray(self, point):
        """Check if point is on the tray"""
        if self.tray is None:
            return False
        return cv2.pointPolygonTest(self.tray, point, False) >= 0
    
    def get_coin_area(self, radius):
        """Calculate coin area in pixels"""
        return np.pi * radius * radius
    
    def _is_large_coin(self, radius):
        """Determine if this is a large coin (5zł) based on radius"""
        if self.tray_area > 0:
            # Calculate coin area as percentage of tray area
            coin_area = self.get_coin_area(radius)
            coin_percentage = (coin_area / self.tray_area) * 100
            return coin_percentage > COIN_SIZE_THRESHOLD
        return radius > 30  # fallback for no tray
    
    def get_tray_area(self):
        """Get the area of the detected tray"""
        return self.tray_area
    
    def get_counts(self):
        """Get detailed coin counts and values"""
        # Count coins by type and location
        small_on_tray = sum(1 for _, _, is_large in self.coins_on_tray if not is_large)
        large_on_tray = sum(1 for _, _, is_large in self.coins_on_tray if is_large)
        small_off_tray = sum(1 for _, _, is_large in self.coins_off_tray if not is_large)
        large_off_tray = sum(1 for _, _, is_large in self.coins_off_tray if is_large)
        
        # Calculate values (5gr = 0.05 PLN, 5zł = 5 PLN)
        value_on_tray = small_on_tray * 0.05 + large_on_tray * 5.0
        value_off_tray = small_off_tray * 0.05 + large_off_tray * 5.0
        total_value = value_on_tray + value_off_tray
        
        return {
            "small_on_tray": small_on_tray,
            "large_on_tray": large_on_tray,
            "small_off_tray": small_off_tray,
            "large_off_tray": large_off_tray,
            "total_on_tray": small_on_tray + large_on_tray,
            "total_off_tray": small_off_tray + large_off_tray,
            "value_on_tray": value_on_tray,
            "value_off_tray": value_off_tray,
            "total_value": total_value
        }

class CoinCounterApp:
    """GUI application for coin counting"""
    
    def __init__(self, image_dir=IMAGE_DIR_PATH):
        self.image_dir = image_dir
        self.images = []
        self.current_index = 0
        self.detector = CoinDetector()
        
        # Load images
        self._load_images()
        
        if not self.images:
            print(f"No images found in {image_dir}")
            return
        
        # Create GUI
        self._setup_gui()
        self.show_image(0)
    
    def _load_images(self):
        """Load images from directory"""
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(self.image_dir, filename))
                if img is not None:
                    self.images.append(img)
    
    def _setup_gui(self):
        """Setup the GUI"""
        self.root = tk.Tk()
        self.root.title("Coin Counter")
        self.root.geometry("1000x700")
        
        # Main display canvas
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls = tk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status bars for coin counts in horizontal layout
        status_frame = tk.Frame(controls)
        status_frame.pack(fill=tk.X, padx=10, pady=2)
        
        # Using a horizontal layout with three columns
        col_width = 33  # Percentage width for each column
        
        # First row - labels
        tk.Label(status_frame, text="ON TRAY", font=("Arial", 9, "bold"), 
                bg="#e6ffe6", width=col_width, bd=1, relief=tk.GROOVE).grid(row=0, column=0, sticky="ew", padx=1)
        tk.Label(status_frame, text="OFF TRAY", font=("Arial", 9, "bold"), 
                bg="#ffebeb", width=col_width, bd=1, relief=tk.GROOVE).grid(row=0, column=1, sticky="ew", padx=1)
        tk.Label(status_frame, text="ALL COINS", font=("Arial", 9, "bold"), 
                bg="#e6f2ff", width=col_width, bd=1, relief=tk.GROOVE).grid(row=0, column=2, sticky="ew", padx=1)
        
        # Second row - values
        self.on_tray_status = tk.StringVar()
        tk.Label(status_frame, textvariable=self.on_tray_status, 
                bg="#e6ffe6", width=col_width, bd=1, relief=tk.GROOVE).grid(row=1, column=0, sticky="ew", padx=1)
        
        self.off_tray_status = tk.StringVar()
        tk.Label(status_frame, textvariable=self.off_tray_status,
                bg="#ffebeb", width=col_width, bd=1, relief=tk.GROOVE).grid(row=1, column=1, sticky="ew", padx=1)
        
        self.all_coins_status = tk.StringVar()
        tk.Label(status_frame, textvariable=self.all_coins_status,
                bg="#e6f2ff", width=col_width, bd=1, relief=tk.GROOVE).grid(row=1, column=2, sticky="ew", padx=1)
        
        # Configure grid column weights
        for i in range(3):
            status_frame.columnconfigure(i, weight=1)
        
        # Image slider
        self.slider = tk.Scale(controls, from_=0, to=len(self.images)-1,
                              orient=tk.HORIZONTAL, 
                              command=lambda v: self.show_image(int(v)))
        self.slider.pack(fill=tk.X, padx=10, pady=5)
        
        # Main status bar at the bottom
        self.status = tk.StringVar()
        tk.Label(controls, textvariable=self.status, 
                bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
        # Key bindings
        self.root.bind("<Left>", lambda e: self.show_image(self.current_index - 1))
        self.root.bind("<Right>", lambda e: self.show_image(self.current_index + 1))
        self.root.bind("<Home>", lambda e: self.show_image(0))
        self.root.bind("<End>", lambda e: self.show_image(len(self.images) - 1))
        
        # Handle resize
        self.canvas.bind("<Configure>", lambda e: self.show_image())
    
    def show_image(self, index=None):
        """Show current image with coin detection"""
        if not self.images:
            return
        
        # Update index if provided
        if index is not None:
            self.current_index = max(0, min(index, len(self.images) - 1))
            self.slider.set(self.current_index)
        
        # Process current image
        image = self.images[self.current_index].copy()
        processed = self.detector.process(image)
        
        # Get canvas dimensions
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        
        # Resize for display
        display_img = self._resize_image(processed, width, height)
        
        # Convert for tkinter
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(width//2, height//2, image=tk_img, anchor=tk.CENTER)
        self.canvas.image = tk_img  # Keep reference to prevent garbage collection
        
        # Get and update coin counts
        counts = self.detector.get_counts()
        
        # Update status bars
        self.on_tray_status.set(
            f"{counts['small_on_tray']} × 5gr + {counts['large_on_tray']} × 5zł = {counts['value_on_tray']:.2f} PLN"
        )
        
        self.off_tray_status.set(
            f"{counts['small_off_tray']} × 5gr + {counts['large_off_tray']} × 5zł = {counts['value_off_tray']:.2f} PLN"
        )
        
        self.all_coins_status.set(
            f"{counts['small_on_tray'] + counts['small_off_tray']} × 5gr + "
            f"{counts['large_on_tray'] + counts['large_off_tray']} × 5zł = {counts['total_value']:.2f} PLN"
        )
        
        # Find coin areas (just get any sample of each type)
        small_area = 0
        large_area = 0
        
        # Check for any small and large coins
        for center, radius, is_large in self.detector.coins_on_tray + self.detector.coins_off_tray:
            if is_large and large_area == 0:
                large_area = self.detector.get_coin_area(radius)
            if not is_large and small_area == 0:
                small_area = self.detector.get_coin_area(radius)
            if small_area > 0 and large_area > 0:
                break
        
        # Update status with tray area and coin areas
        tray_area = self.detector.get_tray_area()
        self.status.set(
            f"Image {self.current_index + 1} of {len(self.images)} | "
            f"Tray Area: {tray_area:.0f} px² | "
            f"5gr Coin Area: {small_area:.1f} px² | "
            f"5zł Coin Area: {large_area:.1f} px² | "
            f"Threshold: {COIN_SIZE_THRESHOLD}% of tray area"
        )
    
    def _resize_image(self, image, width, height):
        """Resize image to fit canvas while preserving aspect ratio"""
        h, w = image.shape[:2]
        aspect = w / h
        
        if height * aspect <= width:
            new_h, new_w = height, int(height * aspect)
        else:
            new_w, new_h = width, int(width / aspect)
        
        return cv2.resize(image, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = CoinCounterApp()
    app.run()