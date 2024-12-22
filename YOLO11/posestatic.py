from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11n-pose.pt")  # load an official model

results = model("mp4/me.mov", save=True, show=True, project="result", name="detection")
#results = model("jpeg/pose.png", save=True, show=True, project="result", name="detection")
