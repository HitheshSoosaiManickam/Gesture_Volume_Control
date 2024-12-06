import cv2 #for cam scanning
import mediapipe as mp #for tracking
import time

class handTracker():
    def __init__(self,mode=False, maxHands=2, detCon=0.5, trackCon=0.5):  
        self.mode = mode
        self.maxHands = maxHands
        self.detCon = detCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        def findHands(self,img):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                    for handLM in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLM.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x*w), int(lm.y*h)
                            print(id, '\n', cx, cy)
                        mpDraw.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)
    
def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0

    while True:
    success,img =  cap.read()
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,70),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), thickness=3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

if __name__ == '__main__':
    main()