import cv2
import torch
import numpy as np
from pathlib import Path
# Load the model
model = torch.hub.load('.', 'custom', 'best.pt', source='local')

# Input video file
video_path = 'video_3.mp4'

# Create a VideoWriter object to save the processed video
output_path = 'output_3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Draw bounding boxes on the frame
    annotated_frame = results.render()[0]

    # Convert the annotated frame to BGR format
    annotated_frame = annotated_frame[:, :, ::-1]

    # Write the frame to the output video
    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
