import cv2
import mediapipe as mp
import time
import PoseDetection.PoseModule as pm
import numpy as np
from playsound import playsound
from threading import Thread

########################
wCam, hCam = 640, 480
########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
cTime = 0
angle = 0
per = 0
bar = 150
counter = 0
switcher = 0
tmp = 0
old_per = 0


def FULL_RANGE_OF_MOTION():
    playsound("C:/Users/chris/PycharmProjects/AdvancedComputerVisionEC/PoseDetection/FULL_RANGE_OF_MOTION.wav")
    global tmp
    tmp = 0

detector = pm.poseDetector()


while True:
    success, img = cap.read()
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        if lmList[14][3] < lmList[13][3]:
            angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, [45, 150], [100, 0])
            bar = np.interp(angle, [45, 150], [150, 400])
            cv2.putText(img, str(int(per)), (lmList[14][1] - 100, lmList[14][2] + 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        # elif np.abs(lmList[14][3] - lmList[13][3]) < 0.2:
        #     detector.findAngle(img, 11, 13, 15)
        #     detector.findAngle(img, 12, 14, 16)
        else:
            angle = detector.findAngle(img, 11, 13, 15)
            per = np.interp(angle, [45, 150], [100, 0])
            bar = np.interp(angle, [45, 150], [150, 400])
            cv2.putText(img, str(int(per)), (lmList[13][1] - 100, lmList[13][2] + 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    if per == 0 or per == 100:
        cv2.rectangle(img, (550, 150), (600, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (550, int(bar)), (600, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (550, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.rectangle(img, (550, 150), (600, 400), (255, 0, 255), 3)
        cv2.rectangle(img, (550, int(bar)), (600, 400), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (550, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
    if per == 100 and switcher == 0:
        switcher = 1
    if switcher == 1 and per == 0:
        switcher = 0
        counter += 1
    if counter < 10:
        cv2.rectangle(img, (0, 480), (200, 280), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(counter)), (50, 430), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 5)
    else:
        cv2.rectangle(img, (0, 480), (200, 280), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(counter)), (0, 430), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255), 5)
    if (old_per - per) > 0 and switcher == 0 and tmp == 0:
        tmp = 1
        new_thread = Thread(target=FULL_RANGE_OF_MOTION)
        new_thread.start()

    if int(per) < 85 and old_per == 1:
        if switcher == 0 and tmp == 0:
            tmp = 1
            new_thread = Thread(target=FULL_RANGE_OF_MOTION)
            new_thread.start()
        old_per = 0

    if int(per) > 90:
        old_per = 1
    else:
        old_per = 0




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)