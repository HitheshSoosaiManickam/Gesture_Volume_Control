import cv2
import time
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

detector = htm.handTracker(detCon=0.7)

while True:
        success,img =  cap.read()
        img = detector.findHands(img)
        lmList = detector.findPos(img, draw=False)
        if len(lmList)!=0:
             #print(lmList[4],lmList[8])

             x1, y1 = lmList[4][1], lmList[4][2]
             x2, y2 = lmList[8][1], lmList[8][2]
             cx, cy = (x1+x2)/2, (y1+y2)/2 

             cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
             cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
             cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
             cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

             length = math.hypot(x2-x1,y2-y1)
             print(length)

             if length < 50:
               cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED )

        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f"FPS: {str(int(fps))}", (10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), thickness=2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)