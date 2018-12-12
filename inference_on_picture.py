from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import cv2
import sys
import numpy as np

path = sys.argv[1]

image = cv2.imread(path)
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = MTCNN()
faces = detector.detect_faces(image)

model = load_model('best_weights.h5')


for face in faces:
    if face['confidence']<0.95:
        continue

    x,y,w,h = face['box']
    padding = 5
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)
    cropped_gray_face = image_grayscale[y-padding:y+h+padding, x-padding:x+w+padding]
    old_x = cropped_gray_face.shape[1]
    old_y = cropped_gray_face.shape[0]
    cropped_gray_face = cv2.resize(cropped_gray_face, (96, 96))
    cropped_gray_face = cropped_gray_face.reshape((96, 96, 1))
    keypoints = model.predict(np.array([cropped_gray_face/255.]))*48+48
    keypoints = keypoints[0].reshape(15, 2)
    for keypoint in keypoints:
        cv2.circle(image, (int(x-padding+keypoint[0]*old_x/96), int(y-padding+keypoint[1]*old_y/96)), 1, (0, 255, 0), 2)

cv2.imshow('result', image)
cv2.waitKey(0)