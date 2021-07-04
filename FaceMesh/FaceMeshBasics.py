import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for id, lm in enumerate(results.multi_face_landmarks):
            mpDraw.draw_landmarks(img, lm, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

            for lms in lm.landmark:
                h, w, c = img.shape
                cx, cy = int(lms.x * w), int(lms.y * h)
                # if id == 0:
                cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)