import cv2
import time
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

# Initialize camera settings
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevTime = 0

# Initialize hand tracker
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize system volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volPer = 0
volBar = 400

# Gesture smoothing variables
smooth_positions = []
smooth_n = 3  # Reduced smoothing window for responsiveness

# Volume smoothing variables
smooth_vol = []
smooth_vol_n = 5  # Number of frames to average for volume smoothing

# Dynamic threshold variables
dynamic_threshold = 50

# Volume bar region (x1, y1, x2, y2)
VOL_BAR_X1, VOL_BAR_Y1 = 300, 80  # Top-left corner of the volume bar
VOL_BAR_X2, VOL_BAR_Y2 = 350, 470  # Bottom-right corner of the volume bar
ROI_PADDING = 50  # Padding around the volume bar for gesture detection

# Function to smooth positions using moving average
def smooth_position(new_position):
    smooth_positions.append(new_position)
    if len(smooth_positions) > smooth_n:
        smooth_positions.pop(0)
    return np.mean(smooth_positions, axis=0)

# Function to smooth volume using moving average
def smooth_volume(new_vol):
    smooth_vol.append(new_vol)
    if len(smooth_vol) > smooth_vol_n:
        smooth_vol.pop(0)
    return np.mean(smooth_vol)

# Function to calculate dynamic threshold based on hand size
def calculate_dynamic_threshold(lmList):
    wrist = lmList[0]
    index_tip = lmList[8]
    hand_size = math.hypot(wrist[1] - index_tip[1], wrist[2] - index_tip[2])
    return hand_size * 0.2  # Smaller threshold multiplier

# Function to check if the hand is in a fist gesture
def is_fist(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks[tip][2] < landmarks[mcp][2]:
            return False
    return True

# Function to check if a point is within the ROI
def is_in_roi(x, y):
    return (VOL_BAR_X1 - ROI_PADDING <= x <= VOL_BAR_X2 + ROI_PADDING) and \
           (VOL_BAR_Y1 - ROI_PADDING <= y <= VOL_BAR_Y2 + ROI_PADDING)

# Main loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    currentVol = volume.GetMasterVolumeLevelScalar() * 100
    volBar = np.interp(currentVol, [0, 100], [470, 80])

    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLM.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                thumb_tip = np.array([lmList[4][1], lmList[4][2]])
                index_tip = np.array([lmList[8][1], lmList[8][2]])
                thumb_tip_smooth = smooth_position(thumb_tip)
                index_tip_smooth = smooth_position(index_tip)

                if is_fist(lmList):
                    volume.SetMute(1, None)
                else:
                    volume.SetMute(0, None)

                    # Check if thumb and index are within the ROI
                    if is_in_roi(thumb_tip_smooth[0], thumb_tip_smooth[1]) and is_in_roi(index_tip_smooth[0], index_tip_smooth[1]):
                        dynamic_threshold = calculate_dynamic_threshold(lmList)
                        length = math.hypot(index_tip_smooth[0] - thumb_tip_smooth[0], index_tip_smooth[1] - thumb_tip_smooth[1])

                        # Adjust length range for smoother volume changes
                        MIN_LENGTH = 20  # Minimum distance for volume control
                        MAX_LENGTH = 100  # Maximum distance for volume control
                        length = np.clip(length, MIN_LENGTH, MAX_LENGTH)  # Clamp values

                        # Map length to volume percentage
                        volPer = np.interp(length, [MIN_LENGTH, MAX_LENGTH], [0, 100])

                        # Smooth the volume percentage
                        volPer = smooth_volume(volPer)

                        # Map volume percentage to system volume
                        vol = np.interp(volPer, [0, 100], [minVol, maxVol])
                        volume.SetMasterVolumeLevel(vol, None)

                #mpDraw.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)

    # Draw UI
    cv2.rectangle(img, (VOL_BAR_X1, VOL_BAR_Y1), (VOL_BAR_X2, VOL_BAR_Y2), (0, 0, 0), 3)
    cv2.rectangle(img, (VOL_BAR_X1, int(volBar)), (VOL_BAR_X2, VOL_BAR_Y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f"Vol: {int(volPer)}%", (40, 520), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img, f"Sys Vol: {int(currentVol)}%", (40, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    # FPS display
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)

    cv2.imshow('Gesture Volume Control', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()