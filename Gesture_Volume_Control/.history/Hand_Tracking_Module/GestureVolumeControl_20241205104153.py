import cv2
import time
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

detector = htm.handTracker(detCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = volPer =0
volBar = 400
while True:
        success,img =  cap.read()
        img = detector.findHands(img)
        lmList = detector.findPos(img, draw=False)
        if len(lmList)!=0:
             #print(lmList[4],lmList[8])

             x1, y1 = lmList[4][1], lmList[4][2]
             x2, y2 = lmList[8][1], lmList[8][2]
             cx, cy = (x1+x2)//2, (y1+y2)//2 

             cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
             cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
             cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
             cv2.circle(img, (cx,cy), 12, (0,0,255), cv2.FILLED)

             length = math.hypot(x2-x1,y2-y1)
             #print(length)

             # Hand Range : 20 to 200
             # Volume Range : -65 to 0

             vol = np.interp(length, [20,200], [minVol,maxVol])
             volBar = np.interp(length, [20,200], [400,150])
             volPer = np.interp(length, [20,200], [0,100])
             volume.SetMasterVolumeLevel(vol, None)

             if length < 25:
               cv2.circle(img, (cx,cy), 12, (0,255,0), cv2.FILLED )

        cv2.rectangle(img, (50,150), (85,400), (0,0,0), 3)
        cv2.rectangle(img, (50,int(volBar)), (85,400), (0,0,0), cv2.FILLED)
        cv2.putText(img, f"FPS: {str(int(volPer))} %", (40,450),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), thickness=2)

        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, f"FPS: {str(int(fps))}", (10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), thickness=2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)