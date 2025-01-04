from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# Load a pre-trained YOLO model
model = YOLO("yolo11l.pt")

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Define video writer for saving the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

log_file = open("Log.txt", "w")

print("Press 'q' to quit")

person_detected = False
last_person_detected_time = None
recorded_frames = []
video_counter = 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        log_file.write("Failed to read frame from webcam.\n")
        continue

    try:
        results = model(frame)
    except Exception as e:
        log_file.write(f"Error during model prediction: {e}\n")
        break

    current_person_detected = False
    for result in results:
        boxes = result.boxes
        detected_classes = [result.names[int(box.cls.item())] for box in boxes] if boxes else []
        log_file.write(f"Detected classes: {detected_classes}\n")
        if 'person' in detected_classes:
            current_person_detected = True

    if current_person_detected:
        print("Person detected.")
        out.write(frame)
        recorded_frames.append(frame)
        last_person_detected_time = time.time()
    else:
        print("No person detected.")
        if last_person_detected_time and time.time() - last_person_detected_time > 4:
            if recorded_frames:
                video_filename = f"output{video_counter}.mp4"
                while os.path.exists(video_filename):
                    video_counter += 1
                    video_filename = f"output{video_counter}.mp4"

                temp_out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                for recorded_frame in recorded_frames:
                    temp_out.write(recorded_frame)
                temp_out.release()

                print(f"Saved video: {video_filename}")
                recorded_frames = []

    try:
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Detection', annotated_frame)
    except Exception as e:
        log_file.write(f"Error displaying frame: {e}\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
log_file.close()
print("Program ended.")
