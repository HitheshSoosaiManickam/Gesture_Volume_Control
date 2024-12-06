import cv2 #for cam scanning
import mediapipe as mp #for tracking
import time

cap = cv2.VideoCapture(0) #opens and works with default camera

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    success,img =  cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cv2.imshow('Image', img)
    cv2.waitKey(1)