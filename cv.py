import numpy as np
import cv2
import pickle

# Learning how to use the OpenCV library for AI face recognition
# Author: Daniel Barahona 
# June 2022
# This code is just a test

LABELS_FILENAME = "train/labels.pickle"
TRAIN_FILENAME = "train/train.yml"

face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAIN_FILENAME)

eye_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_eye.xml')

labels = {'person': 1}
with open(LABELS_FILENAME, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

i=0

while True:
    # Capture frame
    _, frame = cap.read()

    # adjusted = cv2.convertScaleAbs(frame, alpha=2.0, beta=0) # contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        # print(f'ROI:\t{x}\t{y}\t{w}\t{h}')
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # face recognizer model
        id_, confidence = recognizer.predict(roi_gray)
        if confidence >= 45 and confidence <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, labels[id_], (x,y-15), font, 1, (255,255,255), 2)

        # img_item = f"faces/dani/{i}.png"
        # cv2.imwrite(img_item, roi_color)
        # i += 1

        # Draw a rectangle on top of the face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

        # In each face, detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            radius = int(round((ew + eh)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (0,255,0 ), 4)


    # Display frame
    cv2.imshow('face detector', frame)

    # Quit key
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
