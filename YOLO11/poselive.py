from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model(1, show=True)
for result in results:
    boxes = result.boxes
    classes = result.names