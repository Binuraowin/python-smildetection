import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    seccessful_frame_read, frame = webcam.read()
    if not seccessful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)

    cv2.imshow('Smile Detector',frame)

    cv2.waitKey(1)