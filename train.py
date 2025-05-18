import cv2
import numpy as np
from PIL import Image
import os

# Load pre-trained face detector (Haar Cascade)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# Create the LBPH face recognize
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("\n [INFO] Training faces...")
faces, ids = getImagesAndLabels('dataset')

# Train the recognizer
recognizer.train(faces, np.array(ids))

# Save the model
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.save('trainer/trainer.yml')
print(f"\n [INFO] {len(np.unique(ids))} faces trained.")
