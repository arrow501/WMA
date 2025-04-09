import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Globals
IMAGE_DIR_PATH = "CW05-CoinCounter/pliki"
COIN_SIZE_THRESHOLD = 1.8  # Percentage of tray area

class CoinDetector:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.coins_on_tray = []  # [(center, radius, is_large)]
        self.coins_off_tray = []  # [(center, radius, is_large)]
        self.tray = None
        self.tray_area = 0
    
    def process(self, image):
        self.reset()
        result = image.copy()
        
        # Generate weighted grayscale for tray detection
        b, g, r = cv2.split(image)
        weighted_gray = (r.astype(np.float32) * 1.0 + g.astype(np.float32) * -1.0 + 
                         b.astype(np.float32) * -0.9) * 3.0
        weighted_gray = np.clip(weighted_gray, 0, 255).astype(np.uint8)
        
        # Detect tray and on-tray coins
        processed_tray = self._enhance_image(cv2.cvtColor(weighted_gray, cv2.COLOR_GRAY2BGR))
        gray_tray = cv2.cvtColor(processed_tray, cv2.COLOR_BGR2GRAY)
        self.tray = self._detect_tray(weighted_gray)
        
        if self.tray is not None:
            self.tray_area = cv2.contourArea(self.tray)
            cv2.drawContours(result, [self.tray], 0, (255, 0, 0), 2)  # Blue for tray
        
        # Detect coins on tray
        on_tray_circles = cv2.HoughCircles(gray_tray, cv2.HOUGH_GRADIENT, dp=1, 
                                          minDist=36, param1=51, param2=38, 
                                          minRadius=10, maxRadius=0)
        
        # Detect coins off-tray (including coins and non-coins within tray area)
        processed_all = self._enhance_image(image)
        gray_all = cv2.cvtColor(processed_all, cv2.COLOR_BGR2GRAY)
        all_circles = cv2.HoughCircles(gray_all, cv2.HOUGH_GRADIENT, dp=1, 
                                      minDist=36, param1=51, param2=38, 
                                      minRadius=10, maxRadius=0)
        
        # Process on-tray coins
        if on_tray_circles is not None:
            for circle in np.uint16(np.around(on_tray_circles))[0]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                is_large = self._is_large_coin(radius)
                self.coins_on_tray.append((center, radius, is_large))
                
                # Draw coin (green for large/5zł, light blue for small/5gr)
                color = (0, 255, 0) if is_large else (255, 191, 0)
                cv2.circle(result, center, radius, color, 2)
                cv2.circle(result, center, 2, color, 3)
        
        # Process off-tray coins
        if all_circles is not None and self.tray is not None:
            for circle in np.uint16(np.around(all_circles))[0]:
                center = (int(circle[0]), int(circle[1]))
                
                # Only include coins not on tray
                if not self._is_on_tray(center):
                    radius = int(circle[2])
                    is_large = self._is_large_coin(radius)
                    self.coins_off_tray.append((center, radius, is_large))
                    
                    # Draw coin (red for large/5zł, yellow for small/5gr)
                    color = (0, 0, 255) if is_large else (0, 255, 255)
                    cv2.circle(result, center, radius, color, 2)
                    cv2.circle(result, center, 2, color, 3)
        
        return result
    
    def _enhance_image(self, image):
        # Enhance contrast
        img_float = image.astype(np.float32) / 255.0
        img_float = 0.5 + 2.0 * (img_float - 0.5)
        enhanced = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        
        # Apply blur and morphology
        enhanced = cv2.GaussianBlur(enhanced, (11, 11), 0)
        kernel = np.ones((11, 11), np.uint8)
        return cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    def _detect_tray(self, gray):
        # Threshold and clean up with morphology
        _, thresh = cv2.threshold(gray, 200, 200, cv2.THRESH_BINARY)
        kernel = np.ones((17, 17), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None
    
    def _is_on_tray(self, point):
        return self.tray is not None and cv2.pointPolygonTest(self.tray, point, False) >= 0
    
    def get_coin_area(self, radius):
        return np.pi * radius * radius
    
    def get_tray_area(self):
        return self.tray_area
    
    def _is_large_coin(self, radius):
        if self.tray_area > 0:
            coin_percentage = (self.get_coin_area(radius) / self.tray_area) * 100
            return coin_percentage > COIN_SIZE_THRESHOLD
        return radius > 30  # fallback
    
    def get_counts(self):
        # Count coins by type and location
        small_on_tray = sum(1 for _, _, is_large in self.coins_on_tray if not is_large)
        large_on_tray = sum(1 for _, _, is_large in self.coins_on_tray if is_large)
        small_off_tray = sum(1 for _, _, is_large in self.coins_off_tray if not is_large)
        large_off_tray = sum(1 for _, _, is_large in self.coins_off_tray if is_large)
        
        # Calculate values monetarily
        value_on_tray = small_on_tray * 0.05 + large_on_tray * 5.0
        value_off_tray = small_off_tray * 0.05 + large_off_tray * 5.0
        
        return {
            "small_on_tray": small_on_tray,
            "large_on_tray": large_on_tray,
            "small_off_tray": small_off_tray,
            "large_off_tray": large_off_tray,
            "total_on_tray": small_on_tray + large_on_tray,
            "total_off_tray": small_off_tray + large_off_tray,
            "value_on_tray": value_on_tray,
            "value_off_tray": value_off_tray,
            "total_value": value_on_tray + value_off_tray
        }

class CoinCounterApp:
    def __init__(self, image_dir=IMAGE_DIR_PATH):
        self.image_dir = image_dir
        self.detector = CoinDetector()
        self.current_index = 0
        
        # Load images
        self.images = []
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(self.image_dir, filename))
                if img is not None:
                    self.images.append(img)
        
        if not self.images:
            print(f"No images found in {image_dir}")
            return
        
        # Create GUI
        self._setup_gui()
        self.show_image(0)
    
    def _setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Coin Counter")
        self.root.geometry("1000x700")
        
        # Main canvas
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls = tk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status bars in grid layout
        status_frame = tk.Frame(controls)
        status_frame.pack(fill=tk.X, padx=10, pady=2)
        
        # Define column styles
        headers = ["ON TRAY", "OFF TRAY", "ALL COINS"]
        backgrounds = ["#e6ffe6", "#ffebeb", "#e6f2ff"]
        
        # Create header row and status variables
        self.status_vars = []
        for i, (header, bg) in enumerate(zip(headers, backgrounds)):
            status_frame.columnconfigure(i, weight=1)
            tk.Label(status_frame, text=header, font=("Arial", 9, "bold"), 
                    bg=bg, bd=1, relief=tk.GROOVE).grid(row=0, column=i, sticky="ew", padx=1)
            
            var = tk.StringVar()
            self.status_vars.append(var)
            tk.Label(status_frame, textvariable=var, bg=bg, 
                    bd=1, relief=tk.GROOVE).grid(row=1, column=i, sticky="ew", padx=1)
        
        # Image slider
        self.slider = tk.Scale(controls, from_=0, to=len(self.images)-1,
                              orient=tk.HORIZONTAL, 
                              command=lambda v: self.show_image(int(v)))
        self.slider.pack(fill=tk.X, padx=10, pady=5)
        
        # Main status bar
        self.status = tk.StringVar()
        tk.Label(controls, textvariable=self.status, 
                bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)
        
        # Key bindings
        for key, delta in [("<Left>", -1), ("<Right>", 1), 
                          ("<Home>", -float('inf')), ("<End>", float('inf'))]:
            self.root.bind(key, lambda e, d=delta: self.show_image(self.current_index + d))
        
        # Handle resize
        self.canvas.bind("<Configure>", lambda e: self.show_image())
    
    def show_image(self, index=None):
        if not self.images:
            return
        
        # Update index if provided
        if index is not None:
            self.current_index = max(0, min(index, len(self.images) - 1))
            self.slider.set(self.current_index)
        
        # Process current image
        image = self.images[self.current_index].copy()
        processed = self.detector.process(image)
        
        # Display processed image
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        display_img = self._resize_image(processed, width, height)
        
        # Convert to tkinter format
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(width//2, height//2, image=tk_img, anchor=tk.CENTER)
        self.canvas.image = tk_img  # Keep reference
        
        # Update status information
        counts = self.detector.get_counts()
        
        # Update status bars
        status_texts = [
            f"{counts['small_on_tray']} × 5gr + {counts['large_on_tray']} × 5zł = {counts['value_on_tray']:.2f} PLN",
            f"{counts['small_off_tray']} × 5gr + {counts['large_off_tray']} × 5zł = {counts['value_off_tray']:.2f} PLN",
            f"{counts['small_on_tray'] + counts['small_off_tray']} × 5gr + "
            f"{counts['large_on_tray'] + counts['large_off_tray']} × 5zł = {counts['total_value']:.2f} PLN"
        ]
        
        for var, text in zip(self.status_vars, status_texts):
            var.set(text)
        
        # Find sample coin areas
        small_area = large_area = 0
        for center, radius, is_large in self.detector.coins_on_tray + self.detector.coins_off_tray:
            if is_large and large_area == 0:
                large_area = self.detector.get_coin_area(radius)
            if not is_large and small_area == 0:
                small_area = self.detector.get_coin_area(radius)
            if small_area > 0 and large_area > 0:
                break
        
        # Update main status bar
        self.status.set(
            f"Image {self.current_index + 1} of {len(self.images)} | "
            f"Tray Area: {self.detector.get_tray_area():.0f} px² | "
            f"5gr Coin Area: {small_area:.1f} px² | "
            f"5zł Coin Area: {large_area:.1f} px² | "
            f"Threshold: {COIN_SIZE_THRESHOLD}% of tray area"
        )
    
    def _resize_image(self, image, width, height):
        h, w = image.shape[:2]
        aspect = w / h
        
        if height * aspect <= width:
            new_h, new_w = height, int(height * aspect)
        else:
            new_w, new_h = width, int(width / aspect)
        
        return cv2.resize(image, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CoinCounterApp()
    app.run()