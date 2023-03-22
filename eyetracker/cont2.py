import cv2
import numpy as np
import Finder
import VideoCapture
import statistics
import matplotlib.pyplot as plt


finder = Finder.FeatureFinder()
videocap = VideoCapture.MyVideoCapture()
face = cv2.imread("face.jpg")
img = face
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
_, threshold = cv2.threshold(blur,25 ,255, cv2.THRESH_BINARY_INV)

while True:
    ret, frame = videocap.get_frame()
    finder.set_image(frame)
    if len(finder.roi) == 0:
        finder.find_face()
    if len(finder.roi != 0):
        face = finder.roi
        blur = cv2.GaussianBlur(face, (7,7), 0)
        _, threshold = cv2.threshold(blur,25 ,255, cv2.THRESH_BINARY_INV)
        contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(img, [cnt], -1, (0,0,255), 1)
            # M = cv2.moments(cnt)
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])

        # cv2.imshow("img", gray)
        cv2.imshow("blur", blur)
        cv2.imshow("t1", threshold)

        # cv2.imshow("img", img)
        # cv2.imshow("threshold", threshold)
        key = cv2.waitKey(30)
        if key == 27:
            break
cv2.destroyAllWindows()
