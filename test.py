import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import os

# Absolute path to the target image
target_image_path = "D:\\1.Permission restricted\\Eye of the Sauron\\target.png"

if not os.path.isfile(target_image_path):
    raise ValueError(f"Target image file does not exist: {target_image_path}")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the desired YOLO model version

# Load the target image and extract face encoding
target_image = face_recognition.load_image_file(target_image_path)
target_encodings = face_recognition.face_encodings(target_image)
if not target_encodings:
    raise ValueError("No face encodings found in the target image")
target_encoding = target_encodings[0]

# Select the camera feed
camera_index = 1  # Change to 1 for the phone camera, 0 for laptop camera
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    raise ValueError(f"Unable to open camera feed with index: {camera_index}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected = False
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance([target_encoding], face_encoding)
        if face_distances[0] < 0.6:  # Adjust the threshold as needed
            detected = True
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            break  # Stop checking after detecting once

    if detected:
        print("Target person detected")

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()