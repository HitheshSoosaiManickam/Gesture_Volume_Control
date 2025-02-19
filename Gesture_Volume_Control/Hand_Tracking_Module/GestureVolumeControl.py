import cv2
import time
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

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
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = volPer = 0
volBar = 400

def mute_system_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(1, None)

def unmute_system_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(0, None)

def is_fist(landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, and Pinky tips
    finger_mcp = [5, 9, 13, 17]  # MCP joints of respective fingers
    
    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks[tip][2] < landmarks[mcp][2]:  # If fingertip is above MCP, it's open
            return False
    return True  # All fingers folded = Fist detected

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally
    img = detector.findHands(img)
    lmList = detector.findPos(img, draw=False)
    
    # Get current system volume percentage
    currentVol = volume.GetMasterVolumeLevelScalar() * 100
    volBar = np.interp(currentVol, [0, 100], [470, 80])  # Adjust bar fill based on system volume
    
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if 450 <= x1 <= 525 and 80 <= y1 <= 470 and 450 <= x2 <= 525 and 80 <= y2 <= 470:
            if is_fist(lmList):
                mute_system_volume()  # Mute the volume
            else:
                unmute_system_volume()  # Unmute the volume
                length = math.hypot(x2 - x1, y2 - y1)
                vol = np.interp(length, [25, 200], [minVol, maxVol])
                volPer = np.interp(length, [25, 200], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)
    
    # Draw shifted and larger volume bar
    cv2.rectangle(img, (450, 80), (525, 470), (0, 0, 0), 3)
    cv2.rectangle(img, (450, int(volBar)), (525, 470), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f"Vol: {int(volPer)}%", (455, 520), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=2)
    cv2.putText(img, f"Sys Vol: {int(currentVol)}%", (455, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=2)
    
    # FPS Calculation
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=2)
    
    cv2.imshow('Image', img)
    cv2.waitKey(1)



# import cv2
# import time
# import mediapipe as mp
# import numpy as np
# import HandTrackingModule as htm
# import math
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# wCam, hCam = 640, 480

# cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
# prevTime = 0

# detector = htm.handTracker(detCon=0.7)

# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(
#     IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# volume = interface.QueryInterface(IAudioEndpointVolume)
# volRange = volume.GetVolumeRange()
# minVol = volRange[0]
# maxVol = volRange[1]
# vol = volPer = 0
# volBar = 400

# while True:
#     success, img = cap.read()
#     img = cv2.flip(img, 1)  # Flip the image horizontally
#     img = detector.findHands(img)
#     lmList = detector.findPos(img, draw=False)
    
#     # Get current system volume percentage
#     currentVol = volume.GetMasterVolumeLevelScalar() * 100
#     volBar = np.interp(currentVol, [0, 100], [470, 80])  # Adjust bar fill based on system volume
    
#     if len(lmList) != 0:
#         x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
#         x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
#         # Adjusted volume control region (shifted further right and expanded detection area)
#         if 450 <= x1 <= 525 and 80 <= y1 <= 470 and 450 <= x2 <= 525 and 80 <= y2 <= 470:
#             cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)  # Indicate active region
            
#             cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
#             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#             cv2.circle(img, (cx, cy), 12, (0, 0, 255), cv2.FILLED)
            
#             length = math.hypot(x2 - x1, y2 - y1)
#             vol = np.interp(length, [25, 200], [minVol, maxVol])
#             volPer = np.interp(length, [25, 200], [0, 100])
#             volume.SetMasterVolumeLevel(vol, None)
            
#             if length <= 25:
#                 cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
    
#     # Draw shifted and larger volume bar
#     cv2.rectangle(img, (450, 80), (525, 470), (0, 0, 0), 3)
#     cv2.rectangle(img, (450, int(volBar)), (525, 470), (0, 0, 0), cv2.FILLED)
#     cv2.putText(img, f"Vol: {int(volPer)}%", (455, 520), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=2)
#     cv2.putText(img, f"Sys Vol: {int(currentVol)}%", (455, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=2)
    
#     # FPS Calculation
#     currTime = time.time()
#     fps = 1 / (currTime - prevTime)
#     prevTime = currTime
#     cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=2)
    
#     cv2.imshow('Image', img)
#     cv2.waitKey(1)