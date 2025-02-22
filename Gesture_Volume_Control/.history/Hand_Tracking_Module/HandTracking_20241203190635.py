import cv2 #for cam scanning
import mediapipe as mp #for tracking
import time

cap = cv2.VideoCapture(0) #opens and starts capturing with default camera

mpHands = mp.solutions.hands
hands = mpHands.Hands() #for hand tracking specifically
mpDraw = mp.solutions.drawing_utils #for drawing hand points and connections

prevTime = 0
currTime = 0

while True:
    success,img =  cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.imshow('Image', img)
    cv2.waitKey(1)