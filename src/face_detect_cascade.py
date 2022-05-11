import cv2
import numpy as np


frontal_face_cascade = cv2.CascadeClassifier("src\cascades\data\haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontal_faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
    for (x, y, w, h) in frontal_faces:
        print(x,y,w,h)
        roi_color = frame[y:y+h,x:x+w]
        cv2.imwrite("roi_color.png", roi_color)

        color = (0, 0, 255)
        stroke = 2
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)



    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()