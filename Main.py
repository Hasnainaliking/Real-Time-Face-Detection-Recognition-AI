import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
files = os.listdir('dataset')
names = {file.split('.')[1]: file.split('.')[0] for file in files}

cam = cv2.VideoCapture(0)

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        person_name = names.get(str(id),"unknown")

        if (confidence < 60):
            id_str = str(id)
            confidence_str = f" {round(100 - confidence)}%"
        else:
            id_str = "unknown"
            confidence_str = f" {round(100 - confidence)}%"

        cv2.putText(im, str(person_name), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(im, str(confidence_str), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
