import cv2
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk

# Globals
image_dir_path = "CW05-CoinCounter/pliki"


class CoinDetector:
    """Handles coin and tray detection in images"""
    
    def process(self, image):
        """Process image to detect coins and tray"""
        # Reset results
        self.coins_on_tray = []
        self.coins_off_tray = []
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
        
        # 4. Detect on-tray coins
        on_tray_circles = cv2.HoughCircles(
            gray_tray, cv2.HOUGH_GRADIENT, dp=1, minDist=36,
            param1=51, param2=38, minRadius=10, maxRadius=0
        )
        
        # SECOND PIPELINE: Detect all coins, filter to get off-tray coins
        # 5. Process original image without weighted grayscale
        processed_all = self._enhance_image(image)
        gray_all = cv2.cvtColor(processed_all, cv2.COLOR_BGR2GRAY)
        
        # 6. Detect all coins
        all_circles = cv2.HoughCircles(
            gray_all, cv2.HOUGH_GRADIENT, dp=1, minDist=36,
            param1=51, param2=38, minRadius=10, maxRadius=0
        )
        
        # 7. Mark on-tray coins
        if on_tray_circles is not None:
            on_tray_circles = np.uint16(np.around(on_tray_circles))
            for circle in on_tray_circles[0, :]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                self.coins_on_tray.append((center, radius))
                
                # Draw in green
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 2, (0, 255, 0), 3)
        
        # 8. Filter and mark off-tray coins
        if all_circles is not None and self.tray is not None:
            all_circles = np.uint16(np.around(all_circles))
            for circle in all_circles[0, :]:
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])
                
                # Only include coins not on tray
                if not self._is_on_tray(center):
                    self.coins_off_tray.append((center, radius))
                    
                    # Draw in red
                    cv2.circle(result, center, radius, (0, 0, 255), 2)
                    cv2.circle(result, center, 2, (0, 0, 255), 3)
        
        # 9. Draw tray contour
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
    
    def get_counts(self):
        """Get coin counts"""
        return len(self.coins_on_tray), len(self.coins_off_tray)

class CoinCounterApp:
    """GUI application for coin counting"""
    
    def __init__(self, image_dir):
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
        
        # Navigation buttons
        tk.Button(controls, text="Previous", 
                 command=lambda: self.show_image(self.current_index - 1)).pack(side=tk.LEFT, padx=10)
        tk.Button(controls, text="Next", 
                 command=lambda: self.show_image(self.current_index + 1)).pack(side=tk.RIGHT, padx=10)
        
        # Image slider
        self.slider = tk.Scale(controls, from_=0, to=len(self.images)-1,
                              orient=tk.HORIZONTAL, 
                              command=lambda v: self.show_image(int(v)))
        self.slider.pack(fill=tk.X, padx=10, pady=5)
        
        # Status bar
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
        
        # Update status
        on_tray, off_tray = self.detector.get_counts()
        self.status.set(
            f"Image {self.current_index + 1} of {len(self.images)} | "
            f"Coins on tray: {on_tray} | Coins off tray: {off_tray}"
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
    app = CoinCounterApp(image_dir_path)
    app.run()