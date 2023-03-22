import cv2
import numpy as np
import Finder
import VideoCapture
import time

def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    image = cv2.imread("white.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh_val = 180
    start_time = time.time()
    im = image.copy()
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    found = False
    areas = []
    good_contours = []
    prev = []
    for i in range(0, thresh_val):
        im = image.copy()
        print(len(good_contours))
        _, threshold = cv2.threshold(blur, thresh_val - i, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 6:
            cur_area = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if  M["m00"] == 0:
                    break
                cx = int(M["m10"] / M["m00"])
                area = cv2.contourArea( cv2.convexHull(cnt))
                if(area == 0):
                    break
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cur_area.append((cx, cv2.contourArea(cnt),cy))
            if(len(cur_area) < 6):
                continue
            cur_area.sort()
            if(found == False):
                areas = cur_area
                found = True
                continue
            else:
                greater = True
                for j in range(0,6):
                    greater = greater & (cur_area[j][1] >= areas[j][1])
                    if greater == False:
                        break
                if(greater == False):
                    break
                found = True
                areas = cur_area
                good_contours = contours
                # cv2.drawContours(im,good_contours,-1,(255,0,0),1)
                # cv2.imshow('img',im)

        if(len(contours) != 6 and found):
            break
    if(len(good_contours) == 0):
        print("no good contours")
    while True:
        cv2.drawContours(image,good_contours,-1,(255,0,0),1)
        cv2.imshow('img',image)
        if cv2.waitKey(10) & 0xFF == ord('r'):
            break
    order = []
    for cnt in good_contours:
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        order.append((cx,cnt))
    # order.sort()
    # lp = order[1][1]
    # rp = order[4][1]
    print("--- %s seconds ---" % (time.time() - start_time))
    # cv2.drawContours(im, good_contours, -1, (0,255,0), 1)
    # while True:
    #     cv2.imshow('image', im)
    #     if cv2.waitKey(10) & 0xFF == ord('r'):
    #         break

    cv2.namedWindow('t')
    cv2.namedWindow('img')
    # cv2.namedWindow('right')
    cv2.createTrackbar('threshold', 't', 0, 255, nothing)
    # # vid = VideoCapture.MyVideoCapture(0)
    # # finder = Finder.FeatureFinder()
    i2 = cv2.imread("weye.jpg")
    # cv2.fillPoly(i2, [lp], (10, 10, 10))
    # cv2.fillPoly(i2, [rp], (10, 10, 10))
    gray = cv2.cvtColor(i2,cv2.COLOR_BGR2GRAY)
    while True:
        # ret, frame = vid.get_frame()
        im = i2.copy()
        thresh_val = cv2.getTrackbarPos('threshold', 't')
        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, threshold = cv2.threshold(gray, thresh_val ,255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)
        cv2.drawContours(im, hull_list, -1, (0,255,0), 1)
        cv2.imshow('img', im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
