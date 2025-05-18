import cv2
import os

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#mobile_cascade = cv2.CascadeClassifier('haarcascade_mobile_phone.xml')
# Directory to save the images
save_dir = "dataset"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

name = input("Enter user Name: ")
id = input("Enter user ID: ")

count = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #mobile = mobile_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(os.path.join(save_dir, f"{name}.{id}.{count}.jpg"), gray[y:y+h, x:x+w])

    cv2.imshow("Face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 100:  # Take 100 face samples and stop
        break

cam.release()
cv2.destroyAllWindows()
