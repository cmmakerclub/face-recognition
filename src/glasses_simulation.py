import cv2
import numpy as np

from utils import image_resize

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
eyes_cascade = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')
glasses = cv2.imread('images/things/glasses.png', -1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+h] # rectangle
        roi_color = frame[y:y+h, x:x+h]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        eyes_detection = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (ex, ey, ew, eh) in eyes_detection:
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses = image_resize(glasses, width=ew)

            gw, gh, gc = glasses.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    # print(glasses[i, j])
                    if glasses[i, j][3] != 0:
                        roi_color[ey + i, ex + j] = glasses[i, j]

        nose_detection = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose_detection:
            # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_eyes = roi_gray[ny: ny + nh, nx: nx + nw]

    # display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    cv2.imshow('Glasses_simulation', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
