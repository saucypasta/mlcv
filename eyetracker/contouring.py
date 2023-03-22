import cv2
import numpy as np
import Finder
import VideoCapture
import statistics
import matplotlib.pyplot as plt
import math


def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# finder = Finder.FeatureFinder()
# videocap = VideoCapture.MyVideoCapture()
#

img = cv2.imread("glare.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
_, threshold = cv2.threshold(blur, 40,255, cv2.THRESH_BINARY_INV)


contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
(rows, cols) = blur.shape
centerx = cols/2
centery = rows/2

center = None
center_cnt = None
for cnt in contours:
#     M = cv2.moments(cnt)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     if center == None:
#         center_cnt = cnt
#         center = [cX, cY]
#         continue
#     dist1 = distance(center[0], center[1], centerx, centery)
#     dist2 = distance(cX,cY,centerx,centery)
#     if dist2 < dist1:
#         center_cnt= cnt
#         center = [cX, cY]
# if len(contours) != 0:
    cv2.drawContours(img, [cnt], 0, (0,0,255), 1)



cv2.imshow("left eye", blur)
cv2.imshow("blur", gray)
cv2.imshow("t1", threshold)

cv2.imshow("img", img)
# cv2.imshow("threshold", threshold)

while True:
    key = cv2.waitKey(30)
    if key == 27:
        break
cv2.destroyAllWindows()
