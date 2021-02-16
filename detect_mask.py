# importing modules/packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	faces = []
	locs = []
	preds = []

	# create a blob for each video frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob to the trained model detect if faces exist
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (probability)
		confidence = detections[0, 0, i, 2]

		# if probability is less than 50%, its not a face
		if confidence > 0.5:
			# get the x,y coordinates of the face and build a box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Make sure a full face is being shown
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# make sure face exists before predicting
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)

# load the trained face detection model
print("[*] Loading face detection model...")
prototxtPath = os.path.sep.join(["models", "deploy.prototxt"])
weightsPath = os.path.sep.join(["models", "face.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the trained mask detector model
print("[*] loading face mask detector model...")
maskPath = os.path.sep.join(["models", "mask.model"])
maskNet = load_model(maskPath)

# start the video stream
print("[*] starting video stream...")
vs = VideoStream(src=0).start()
# sleep since some cameras are slow to start. Increase this if you face any errors
time.sleep(2)

# for every frame in the video, the following is executed
while True:
	# grab the current frame
	frame = vs.read()
	# resize the frame to 400 pixels
	frame = imutils.resize(frame, width=400)

	# detect face as well as if they are wearing a mask
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations
	for (box, pred) in zip(locs, preds):
		# ready the bounding box predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# set the class label and color depending on the prediction
		if mask > withoutMask:
			label = "Mask" 
		else:
			label =  "No Mask"
		# set red if no mask and green if there is a mask
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Add the detection probability
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# print the box
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# display the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF