import cv2
import numpy as np
import HandDetection.HandTrackingModule as htm
import time
import autopy

########################
wCam, hCam = 640, 480
frameReduction = 100
smoothening = 5
########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
cTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.HandDetector(min_detection_confidence=0.7, max_num_hands=1)

wScreen, hScreen = autopy.screen.size()
print(wScreen, hScreen)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:3]
        x2, y2 = lmList[12][1:3]
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameReduction, frameReduction), (wCam - frameReduction, hCam - frameReduction), (255, 0, 255), 3)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, [frameReduction, wCam-frameReduction], [0, wScreen])
            y3 = np.interp(y1, [frameReduction, hCam-frameReduction], [0, hScreen])

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScreen - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocY = clocY
            plocX = clocX


        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
