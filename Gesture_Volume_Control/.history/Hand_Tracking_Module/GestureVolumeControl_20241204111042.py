import cv2
import time
import mediapipe as mp
import numpy as np

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

while True:
        success,img =  cap.read()
        # img = detector.findHands(img)
        # lmList = detector.findPos(img)
        # if len(lmList)!=0:
        #     print(lmList[4])

        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f"FPS: {str(int(fps))}", (10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), thickness=1)

        cv2.imshow('Image', img)
        cv2.waitKey(1)