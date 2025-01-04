import threading
import time
import os
from ultralytics import YOLO
import cv2
import numpy as np

#Detection with YOLO
#Uses a camera input to detect objects
#If at least one of the objects is a person it saves the frames to a video
#If there is no person detected for over 4 seconds, the video is saved locally

def yolo_detection_program():
    model = YOLO("yolo11l.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    person_detected = False
    last_person_detected_time = None
    recorded_frames = []
    video_counter = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        if np.mean(frame) == 0:
            continue

        results = model(frame)

        current_person_detected = False
        for result in results:
            boxes = result.boxes
            detected_classes = [result.names[int(box.cls.item())] for box in boxes] if boxes else []
            if 'person' in detected_classes:
                current_person_detected = True

        if current_person_detected:
            out.write(frame)
            recorded_frames.append(frame)
            last_person_detected_time = time.time()
        else:
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("YOLO Detection Program ended.")

#Action recognition with MMAction2
#Monitoring if there is a new video input
#If there is a new video input, the input is being analysed and evaluated
#If the analysed video displays a blacklisted action a notification to another system is sent

def monitor_directory_program():
    def hello_world_function():
        print("Hello World")

    directory = "."
    existing_files = set(os.listdir(directory))

    while True:
        current_files = set(os.listdir(directory))
        new_files = current_files - existing_files

        for file in new_files:
            if file.startswith("output") and file.endswith(".mp4"):
                print(f"New video detected: {file}")
                hello_world_function()

        existing_files = current_files
        time.sleep(1)

#Thread management
if __name__ == "__main__":
    yolo_thread = threading.Thread(target=yolo_detection_program)
    monitor_thread = threading.Thread(target=monitor_directory_program)

    yolo_thread.start()
    monitor_thread.start()

    yolo_thread.join()
    monitor_thread.join()

    print("Both programs have ended.")
