from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11n.pt")

results = model("jpeg/IMG_8048.jpeg", save=True, show=True, project="result", name="detection")
#results = model("mp4/me.mov", save=True, show=True, project="result", name="detection")
print(dir(results))