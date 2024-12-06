import cv2 
import mediapipe as mp 
import time

class handTracker():
    def __init__(self,mode=False, maxHands=2, detCon=0.5, trackCon=0.5):  
        self.mode = mode
        self.maxHands = maxHands
        self.detCon = detCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
    max_num_hands=self.maxHands,
    min_detection_confidence=self.detCon,
    min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
                for handLM in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, handNo=0, draw =True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0,0,255), cv2.FILLED)

        return lmList

def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0

    detector = handTracker()

    while True:
        success,img =  cap.read()
        img = detector.findHands(img)
        lmList = detector.findPos(img)
        if len(lmList)!=0:
            print(lmList[4])

        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10,70),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), thickness=3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()