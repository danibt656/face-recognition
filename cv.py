import numpy as np
import cv2
import pickle
import dlib

# Learning how to use the OpenCV library for AI face recognition
# Author: Daniel Barahona 
# June 2022
# This code is just a test

LABELS_FILENAME = "train/labels.pickle"
TRAIN_FILENAME = "train/train.yml"

C_GREEN = (0,255,0)
C_BLUE = (255,0,0)
C_RED = (0,0,255)
C_WHITE = (255,255,255)

# OpenCV detector
face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAIN_FILENAME)

# dlib detector
dlib_det = dlib.get_frontal_face_detector()
# landmarks predictor
lmark = dlib.shape_predictor('./train/shape_predictor_68_face_landmarks.dat')

labels = {'person': 1}
with open(LABELS_FILENAME, 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# Convert a boundary box to a rectangle in dlib's format
def bb_to_rect(x, y, w, h):
    left = x
    top = y
    right = w + x
    bottom = h + x

    return dlib.rectangle(left, top, right, bottom)

# Convert a dlib rectangle into a boundary box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)

# Convert a dlib shape into a 68 (x,y) coordinates numpy array
def shape_to_np(shape, dtype='int'):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# main function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        # Capture frame
        _, frame = cap.read()
        frame =  cv2.flip(frame, 1)

        # convert to grayscale & equalize contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # detect faces...
        # ... using opencv cascade detector
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        # for (x,y,w,h) in faces:
        #     rect = bb_to_rect(x,y,w,h)
            
        #     # Draw a rectangle on top of the face
        #     cv2.rectangle(frame, (x,y), (x+w, y+h), C_GREEN, 2)

        #     # Get face region of interest (roi)
        #     roi_gray = gray[y:y+h, x:x+w]

        #     shape = lmark(gray, rect)
        #     shape = shape_to_np(shape)

        #     # draw facial landmarks
        #     for (sx,sy) in shape:
        #         cv2.circle(frame, (sx,sy), 1, C_RED, -1)

        #     # face recognizer model
        #     id_, confidence = recognizer.predict(roi_gray)
        #     if confidence >= 45 and confidence <= 85:
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         text = f'{labels[id_]} {round(confidence, 1)}%'
        #         cv2.putText(frame, text, (x,y-15), font, 1, C_WHITE, 2)
    
        #     # cv2.imwrite(f"faces/dani/{i}.png", roi_color)
        
        # ... using dlib's detector
        faces = dlib_det(gray, 1)
        for rect in faces:
            # Draw a rectangle on top of the face
            (x,y,w,h) = rect_to_bb(rect)
            cv2.rectangle(frame, (x,y), (x+w, y+h), C_GREEN, 2)

            # Get face region of interest (roi)
            roi_gray = gray[y:y+h, x:x+w]

            shape = lmark(gray, rect)
            shape = shape_to_np(shape)

            # draw facial landmarks
            for (sx,sy) in shape:
                cv2.circle(frame, (sx,sy), 1, C_RED, -1)

            # face recognizer model
            id_, confidence = recognizer.predict(roi_gray)
            if confidence >= 45 and confidence <= 85:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'{labels[id_]} {round(confidence, 1)}%'
                cv2.putText(frame, text, (x,y-15), font, 1, C_WHITE, 2)

        # Display frame
        cv2.imshow('face detector', frame)

        # Quit key
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
