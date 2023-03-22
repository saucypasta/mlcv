from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import dlib
import cv2
import VideoCapture
import math
import time
import pyautogui
import win32api
from csv import writer
import Seq
import seqpose

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def pt_distance(coord1, coord2):
    (x1, y1) = coord1
    (x2, y2) = coord2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


class EyeTracker:
    def __init__(self, vid, ear_thresh=.19, pupil_thresh=30, blink_consec=2, seqp = None):
        self.vid = vid
        self.ear_thresh = ear_thresh
        self.pupil_thresh = pupil_thresh
        self.blink_consec = blink_consec
        self.seqp = seqp
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.r_left_corner_ind = 36
        self.r_right_corner_ind = 39
        self.l_left_corner_ind = 42
        self.l_right_corner_ind = 45
        self.left_counter = 0
        self.right_counter = 0
        self.left_blinked = False
        self.right_blinked = False
        self.img = []
        self.face_points = []
        self.data = []
        self.good_data = False
        self.bad_count = 0

    def get_image(self, img):
        return self.img

    def get_faces(self):
        if(self.img == []):
            print("No image")
            return []
        return self.detector(self.img, 0)

    def gen_face_points(self):
        faces = self.get_faces()
        if(faces == []):
            print("No faces")
            return []
        for face in faces:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            self.face_points = self.predictor(self.img, face)
            self.face_points = face_utils.shape_to_np(self.face_points)

    def get_face_points(self):
        if(self.face_points == []):
            print("No face points")
        return self.face_points

    def get_data(self):
        if(not self.good_data):
            return []
        return self.data

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates

        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # A = pt_distance(eye[1], eye[5])
        # B = pt_distance(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # C = pt_distance(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear

    def nothing(self, x):
        pass

    def mainloop(self,save=False):
        self.good_data = False
        ret, frame = self.vid.get_frame()
        self.img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cursor = pyautogui.position()
        cursorx = cursor.x
        cursory = cursor.y
        # cv2.createTrackbar('threshold', 'image', 50, 255, self.nothing)
        if(ret):
            self.gen_face_points()
            if(self.face_points == []):
                return []
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            leftEye = self.face_points[self.lStart:self.lEnd]
            rightEye = self.face_points[self.rStart:self.rEnd]

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # check for blinking
            # left blibk
            if leftEAR < self.ear_thresh and rightEAR >= self.ear_thresh:
                self.left_counter += 1
                return []
            else:
                if self.left_counter >= self.blink_consec:
                    # print("Left Blink")
                    self.left_blinked = True

                # reset the eye frame counter
                self.left_counter = 0

            # right blink
            if rightEAR < self.ear_thresh and leftEAR >= self.ear_thresh:
                self.right_counter += 1
                return []
            else:
                if self.right_counter >= self.blink_consec:
                    # print("Right blink")
                    self.right_blinked = True
                # reset the eye frame counter
                self.right_counter = 0

            black_frame = np.zeros_like(self.img).astype(np.uint8)
            cv2.fillPoly(black_frame , [leftEyeHull], (255, 255, 255))
            cv2.fillPoly(black_frame , [rightEyeHull], (255, 255, 255))
            mask = black_frame == 255
            w_mask = np.array(255 * (mask == 0)).astype(np.uint8)
            sclera = self.img * mask
            pupil = (self.img * mask) + w_mask

            cv2.putText(frame, "REAR: {:.2f}".format(leftEAR), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "LEAR: {:.2f}".format(rightEAR), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            blur = cv2.GaussianBlur(sclera, (5,5), 0)
            thresh_val = 200
            found = False
            areas = []
            good_contours = []
            prev = []
            for i in range(0, thresh_val):
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
                        cur_area.append((cx, cv2.contourArea(cnt)))
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
                if(len(contours) != 6 and found):
                    break

            if(len(good_contours) == 6):
                self.bad_count = 0
            if(len(good_contours) != 6 and self.bad_count < 5):
                self.bad_count = self.bad_count + 1
                return []

            if self.bad_count == 0:
                order = []
                for cnt in good_contours:
                    M = cv2.moments(cnt)
                    cx = int(M["m10"] / M["m00"])
                    order.append([cx,[cnt]])
                order.sort(key=lambda x: x[0])
                lp = order[1][1][0]
                rp = order[4][1][0]
                cv2.fillPoly(pupil, [lp], (10, 10, 10))
                cv2.fillPoly(pupil, [rp], (10, 10, 10))

            blur = cv2.GaussianBlur(pupil, (7,7), 0)
            _, threshold = cv2.threshold(blur, self.pupil_thresh, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            centers = []
            eyes = []
            areas = []
            for cnt in contours:
                hull = cv2.convexHull(cnt)
                area = cv2.contourArea(cnt)
                areas.append(area)
                if(area > 500 or area < 15):
                    # cv2.imshow('image',frame)
                    continue
                cv2.drawContours(frame, [hull], -1, (0,0,255), 1)
                M = cv2.moments(hull)
                if M["m00"] == 0:
                    break
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append([cX, cY])
                eyes.append(cnt)

            if len(centers) != 2:
                # print("Bad center areas: ",areas)
                # cv2.imwrite("bad.jpg",pupil)
                return []
            # first center is right eye (leftmost eye in image)
            left_iris = eyes[1]
            right_iris = eyes[0]
            if centers[0][0] > centers[1][0]:
                tmp = centers[0]
                centers[0] = centers[1]
                centers[1] = tmp
                left_iris = eyes[0]
                right_iris = eyes[1]

            ##############SCLERA RATIOS########
            # rmin_left_dist = 0
            # rmin_right_dist = 0
            # first = True
            # for pt in right_iris:
            #     (x, y) = pt[0]
            #     (xl, yl) = self.face_points[self.r_left_corner_ind]
            #     (xr, yr) = self.face_points[self.r_right_corner_ind]
            #     ldist = distance(x, y, xl, yl)
            #     rdist = distance(x, y, xr, yr)
            #     if(first):
            #         rmin_left_dist = ldist
            #         rmin_right_dist = rdist
            #         first = False
            #         continue
            #     if ldist < rmin_left_dist:
            #         rmin_left_dist = ldist
            #     if rdist < rmin_right_dist:
            #         rmin_right_dist = rdist
            #
            #
            # lmin_left_dist = 0
            # lmin_right_dist = 0
            # first = True
            # for pt in left_iris:
            #     (x, y) = pt[0]
            #     (xl, yl) = self.face_points[self.l_left_corner_ind]
            #     (xr, yr) = self.face_points[self.l_right_corner_ind]
            #     ldist = distance(x, y, xl, yl)
            #     rdist = distance(x, y, xr, yr)
            #     if(first):
            #         lmin_left_dist = ldist
            #         lmin_right_dist = rdist
            #         first = False
            #         continue
            #     if ldist < lmin_left_dist:
            #         lmin_left_dist = ldist
            #     if rdist < lmin_right_dist:
            #         lmin_right_dist = rdist
            # self.dists = [rmin_left_dist, rmin_right_dist, lmin_right_dist, lmin_left_dist]
            ######################################################
            self.centers = centers
            # cv2.circle(frame, tuple(centers[0]), 1, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(centers[1]), 1, (0, 0, 255), -1)
            self.data = [self.centers, rightEAR, leftEAR, self.face_points]
            self.good_data = True
            rowx = []
            pos = []
            for (x,y) in self.face_points:
                pos.append(x/1919)
                pos.append(y/1079)
            p = self.seqp.predict(np.array([pos]))
            x = p[0][0]
            y = p[0][1]
            rowx = [self.centers[0][0]/1919, self.centers[0][1]/1079, self.centers[1][0]/1919, self.centers[1][1]/1079, rightEAR,leftEAR,x,y]
            if(save != 0):
                with open("xtrain2.csv", 'a+', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(rowx)
                rowy = [cursorx/1919, cursory/1079]
                with open("ytrain2.csv", 'a+', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(rowy)
            return [frame, rowx]




def nothing(val):
    pass
# # #
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 42, 255, nothing)
vs = VideoCapture.MyVideoCapture()
sp = seqpose.SEQP()
et = EyeTracker(vid = vs,seqp = sp)
s = Seq.SEQ()
while True:
    thresh_val = cv2.getTrackbarPos('threshold', 'image')
    et.pupil_thresh = thresh_val
    # print(pyautogui.position())
    save = win32api.GetAsyncKeyState(0x20)
    # start_time = time.time()
    f = et.mainloop(save)
    # if len(f) !=0:
    if win32api.GetAsyncKeyState(0x09) and len(f) !=0:
        print(np.shape(f[1]))
        p = s.predict(np.array([f[1]]))
        x = p[0][0] * 1919
        y = p[0][1] * 1079
        pyautogui.moveTo(x,y)
        # print("real ",pyautogui.position().x,pyautogui.position().y)
    if win32api.GetAsyncKeyState(0x02):
        s.init_train()
        print(np.shape(s.x_train))
        s.train_model()
        s.save_weights()

    if(len(f) != 0):
        cv2.imshow('image', f[0])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(.01)
    # print("--- %s seconds ---" % (time.time() - start_time))
cv2.destroyAllWindows()
