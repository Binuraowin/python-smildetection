import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    seccessful_frame_read, frame = webcam.read()

    cv2.imshow('Smile Detector',frame)

    cv2.waitKey(1)