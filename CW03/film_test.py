import cv2
import os

# Save every nth frame (default is 30)
n = 30

# Create output directory if it doesn't exist
output_dir = "./ball_frames"
os.makedirs(output_dir, exist_ok=True)

video = cv2.VideoCapture()
video.open(r'CW03\movingball.mp4')
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

counter = 1

while True:
    success, frame_rgb = video.read()
    if not success:
        break
    print('klatka {} z {}'.format(counter, total_frames))
    
    # Add text to even frames
    if counter % 2 == 0:
        cv2.putText(frame_rgb, 'pilka', (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 4)
    
    # Save every nth frame
    if counter % n == 0:
        frame_filename = os.path.join(output_dir, f"frame_{counter:05d}.jpg")
        cv2.imwrite(frame_filename, frame_rgb)
    
    counter = counter + 1

video.release()
print(f"Saved every {n}th frame to {output_dir}")
