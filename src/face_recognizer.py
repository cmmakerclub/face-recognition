import pickle
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {value:key for key, value in og_labels.items()}

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        # recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_) 
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA) # img, text, org, fontFace, fontSacle, color[,thickness[, lineType[, buttonLeftOrigin]]]

        # write image
        img_item = "current_captured.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) # Blue, Green, Red
        stroke = 2 # Pixel
        end_cord_x = x + w
        enc_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, enc_cord_y), color, stroke)
        eyes_detection = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes_detection:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display
    cv2.imshow('Face_recognizer v0.1', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()