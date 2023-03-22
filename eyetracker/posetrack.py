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
import seqpose

pyautogui.FAILSAFE = False

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
    def __init__(self, vid, ear_thresh=.3, pupil_thresh=30, blink_consec=2):
        self.vid = vid
        self.ear_thresh = ear_thresh
        self.pupil_thresh = pupil_thresh
        self.blink_consec = blink_consec
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
                    print("Left Blink")
                    self.left_blinked = True

                # reset the eye frame counter
                self.left_counter = 0

            # right blink
            if rightEAR < self.ear_thresh and leftEAR >= self.ear_thresh:
                self.right_counter += 1
                return []
            else:
                if self.right_counter >= self.blink_consec:
                    print("Right Blink")
                    # print("Right blink")
                    self.right_blinked = True
                # reset the eye frame counter
                self.right_counter = 0
            cv2.putText(frame, "REAR: {:.2f}".format(leftEAR), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "LEAR: {:.2f}".format(rightEAR), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rowx = []
            for (x,y) in self.face_points:
                rowx.append(x/1919)
                rowx.append(y/1079)
            if(save != 0):
                with open("xpose.csv", 'a+', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(rowx)
                rowy = [cursorx/1919, cursory/1079]
                with open("ypose.csv", 'a+', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(rowy)
            return [frame, rowx]




def nothing(val):
    pass
# # #
cv2.namedWindow('image')
cv2.createTrackbar('threshold', 'image', 42, 255, nothing)
vs = VideoCapture.MyVideoCapture()
et = EyeTracker(vid = vs)
s = seqpose.SEQP()
while True:
    # print(pyautogui.position())
    thresh_val = cv2.getTrackbarPos('threshold', 'image')
    et.ear_thresh = thresh_val/100
    save = win32api.GetAsyncKeyState(0x20)
    # start_time = time.time()
    f = et.mainloop(save)
    if len(f)!=0:
    # if win32api.GetAsyncKeyState(0x09) and len(f) !=0:
        p = s.predict(np.array([f[1]]))
        x = p[0][0] * 1919
        y = p[0][1] * 1079
        pyautogui.moveTo(x,y)
        print("model: ",x,y)
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
