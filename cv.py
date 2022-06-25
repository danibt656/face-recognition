# from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import sleep
import pickle
import dlib
import cv2

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

EAR_THRESH = 0.3
EAR_CONSEC_FRAMES = 3

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

def distance_euclidean(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# calculate the eye aspect ratio given its landmarks
def eye_aspect_ratio(eye):
    A = distance_euclidean(eye[1], eye[5])
    B = distance_euclidean(eye[2], eye[4])
    C = distance_euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# main function
def main():
    blink_counter = 0
    blink_total = 0
    (lStart, lEnd) = (42,48)
    (rStart, rEnd) = (36,42)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--plot", required=False, action="store_true",
        help="show prediction confidence plots after quit (q)")
    args = vars(ap.parse_args())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    sleep(0.2)

    # Plot confidences in each time instant
    # confs is a dict containing each label and the corresponding array of
    # confidences obtained per instant for that label
    t = 0
    tss = []
    confs = {v:[] for _,v in labels.items()}
    
    while True:
        # Capture frame
        _, frame = cap.read()
        frame =  cv2.flip(frame, 1)

        # convert to grayscale & equalize contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # DETECT FACES ...

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
        faces = dlib_det(gray, 0)
        for rect in faces:
            # Draw a rectangle on top of the face
            (x,y,w,h) = rect_to_bb(rect)
            cv2.rectangle(frame, (x,y if y>0 else 0), (x+w, y+h), C_GREEN, 2)

            # Get face region of interest (roi)
            roi_gray = gray[y:y+h, x:x+w]

            # draw facial landmarks
            shape = lmark(gray, rect)
            shape = shape_to_np(shape)
            for (sx,sy) in shape:
                cv2.circle(frame, (sx,sy if sy>0 else 0), 1, C_RED, -1)

            # blink counter
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear_avg = (leftEAR + rightEAR) / 2.0
            if ear_avg < EAR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EAR_CONSEC_FRAMES:
                    blink_total += 1
                blink_counter = 0
            cv2.putText(frame, f'Blinks: {blink_total}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, C_WHITE, 1)
            # draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, C_GREEN, 1)
            cv2.drawContours(frame, [rightEyeHull], -1, C_GREEN, 1)

            faceHull = cv2.convexHull(shape)
            cv2.drawContours(frame, [faceHull], -1, C_WHITE, 1)

            # face recognizer model
            id_, confidence = recognizer.predict(roi_gray)
                # add data to plot's axis
            if args['plot']:
                for label,_ in confs.items():
                    if label == labels[id_]:
                        confs[label].append(confidence)
                    else:
                        confs[label].append(50)
                tss.append(t)
                t += 1
                # display veredict
            text = f'{labels[id_]} {round(confidence, 1)}%'
            cv2.putText(frame, text, (x,y-15 if y-15>0 else 0), cv2.FONT_HERSHEY_SIMPLEX, 1, C_WHITE, 2)

        cv2.imshow('face detector', frame)
        # Quit key
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # close open stuff
    cap.release()
    cv2.destroyAllWindows()

    if args['plot']:
        # confidences/time plot
        _, ax = plt.subplots(2)
        for l,c in confs.items():
            ax[0].plot(tss, c, label=l)
        ax[0].legend(loc='lower right')

        # confidence-per-label plot
        cc = list(confs.values())
        avg_cc = []
        for c in cc:
            avg_cc.append(np.average(c))
        ax[1].bar(list(confs.keys()), avg_cc, width=1, edgecolor='white', linewidth=0.7)
        # show plots
        plt.show()

if __name__ == '__main__':
    main()
