from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model.predict(0, show=True, stream=True, conf=0.8)

data = []
for result in results:
    data.append(result.keypoints)
    with open("data.txt", "w") as f:
        for d in data:
            if len(data) > 0:
                f.write(str(d))
            else:
                f.write("nothing valid to write.")

# for result in results:
#     result.save_crop()

#     Results objects have the following attributes:

# Attribute	Type	Description
# orig_img	numpy.ndarray	The original image as a numpy array.
# orig_shape	tuple	The original image shape in (height, width) format.
# boxes	Boxes, optional	A Boxes object containing the detection bounding boxes.
# masks	Masks, optional	A Masks object containing the detection masks.
# probs	Probs, optional	A Probs object containing probabilities of each class for classification task.
# keypoints	Keypoints, optional	A Keypoints object containing detected keypoints for each object.
# obb	OBB, optional	An OBB object containing oriented bounding boxes.
# speed	dict	A dictionary of preprocess, inference, and postprocess speeds in milliseconds per image.
# names	dict	A dictionary of class names.
# path	str	The path to the image file.
# Results objects have the following methods:

# Method	Return Type	Description
# update()	None	Update the boxes, masks, and probs attributes of the Results object.
# cpu()	Results	Return a copy of the Results object with all tensors on CPU memory.
# numpy()	Results	Return a copy of the Results object with all tensors as numpy arrays.
# cuda()	Results	Return a copy of the Results object with all tensors on GPU memory.
# to()	Results	Return a copy of the Results object with tensors on the specified device and dtype.
# new()	Results	Return a new Results object with the same image, path, and names.
# plot()	numpy.ndarray	Plots the detection results. Returns a numpy array of the annotated image.
# show()	None	Show annotated results to screen.
# save()	None	Save annotated results to file.
# verbose()	str	Return log string for each task.
# save_txt()	None	Save predictions into a txt file.
# save_crop()	None	Save cropped predictions to save_dir/cls/file_name.jpg.
# tojson()	str	Convert the object to JSON format.