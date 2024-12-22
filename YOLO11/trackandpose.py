import cv2
from ultralytics import YOLO

# Load the YOLO model
trackmodel = YOLO("yolo11n.pt")
posemodel = YOLO("yolo11n-pose.pt")

# Open the video file
video_path = "mp4/me.mov"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    #model starten!
    if success:
        poseresults = posemodel.predict(frame)
        # classes=[0] heißt, dass nur objekte der klasse 0 erkannt werden (0=person)
        trackresults = trackmodel.track(frame, classes=[0])

        # Visualize the results on the frame
        annotated_frame = poseresults[0].plot()
        annotated_frame = trackresults[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()