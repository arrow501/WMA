import cv2
import matplotlib as mpl  # Import matplotlib itself for version
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk

print("OpenCV version:", cv2.__version__)
print("Matplotlib version:", mpl.__version__)  # Use matplotlib, not plt
print("NumPy version:", np.__version__)
print("All imports successful!")

# Test a simple OpenCV function
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(test_img, (25, 25), (75, 75), (0, 255, 0), 2)
print("OpenCV drawing successful!")

# Test tkinter
root = tk.Tk()
root.title("Tkinter Test")
label = tk.Label(root, text="Tkinter is working!")
label.pack()
root.after(2000, root.destroy)  # Close window after 2 seconds
root.mainloop()
print("Tkinter test successful!")

# Test matplotlib
plt.figure()
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Test Plot")
plt.show()
print("Matplotlib test successful!")