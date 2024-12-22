from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11l.pt")

# results = model("jpeg/IMG_8048.jpeg", save=True, show=True, project="result", name="detection")
results = model(0, show=True)
for result in results:
    boxes = result.boxes
    classes = result.names