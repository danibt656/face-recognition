import os
import numpy as np
from PIL import Image
import cv2
import pickle

# Browse through image directories with faces
# then train a model to learn recognize those faces

face_cascade = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

LABELS_FILENAME = "train/labels.pickle"
TRAIN_FILENAME = "train/train.yml"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'faces')

curr_id = 0
label_ids = {}
y_labels = []
x_train = []

for root,dirs,files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(' ', '-').lower()

            if not label in label_ids:
                label_ids[label] = curr_id
                curr_id += 1

            pil_image = Image.open(path).convert('L') #grayscale
            final_image = pil_image.resize((550,550), Image.ANTIALIAS)
            image_array = np.array(final_image, 'uint8')

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(label_ids[label])

with open(LABELS_FILENAME, 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save(TRAIN_FILENAME)