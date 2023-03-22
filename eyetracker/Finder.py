import cv2
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

class FeatureFinder:
    def __init__(self, image = [], thresh_val = 30):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.image = image
        self.face = []
        self.face_center = []
        self.roi = None
        self.left_center = []
        self.right_center = []
        self.prev_left_center = []
        self.prev_right_center = []
        self.eyes = []
        self.eyes_found = False
        self.centers_found = False
        self.radius = 24
        self.thresh_img=[]
        self.eye_img = []
        self.thresh_val = thresh_val
        self.find_face()
        self.find_eyes()


    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def find_face(self):
        #no image
        if len(self.image) == 0:
            return 0
        faces = self.face_cascade.detectMultiScale(self.image, 1.3, 5)
        if(len(faces) != 0):
            self.face = faces[0]
        return 1

    def eye_center(self,eye, thresh_save):
        (x, y, w, h) = eye
        img = self.roi[y:y+h, x:x+w]
        blur = cv2.GaussianBlur(img, (7,7), 0)
        _, threshold = cv2.threshold(blur, self.thresh_val,255, cv2.THRESH_BINARY_INV)
        if(thresh_save):
            self.thresh_img = threshold
            self.eye_img = img
        contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        (rows, cols) = blur.shape
        centerx = cols/2
        centery = rows/2

        center = None
        center_cnt = None
        if len(contours) != 0:
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if center == None:
                    center_cnt = cnt
                    center = [cX, cY]
                    continue
                dist1 = distance(center[0], center[1], centerx, centery)
                dist2 = distance(cX,cY,centerx,centery)
                if dist2 < dist1:
                    center_cnt= cnt
                    center = [cX, cY]
        if center == None:
            return []
        return [center[0] + x, center[1] + y]

    def find_eyes(self):
        self.eyes_found = False
        if len(self.face) == 0:
            return 0
        (x, y, w, h) = self.face
        self.roi = self.image[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(self.roi)
        if len(eyes) == 2:
            (x1, y1, w1, h1) = eyes[0]
            (x2, y2, w2, h2) = eyes[1]
            left_eye = eyes[0]
            right_eye = eyes[1]
            if x2 < x1:
                left_eye = eyes[1]
                right_eye = eyes[0]
            self.eyes = [left_eye, right_eye]
            self.eyes_found = True
            return 1
        return 0

    def get_eye_center(self):
        self.centers_found = False
        if self.eyes_found:
            self.left_center = self.eye_center(left_eye,False)
            if(self.left_center == []):
                return 0
            self.right_center = self.eye_center(right_eye,False)
            if(self.right_center == []):
                return 0
        self.centers_found = True


    def get_eyes(self, image):
        self.set_image(image)
        if(self.find_face() == 0):
            return []
        if(self.find_eyes() == 0):
            return []
        return self.eyes

    def get_eye_locations(self):
        if(self.eyes_found):
            return [self.left_center, self.right_center]
        print("no eyes")
        return []


    def draw_face_boundary(self):
        if len(self.face) == 0:
            return
        (x, y, w, h) = self.face
        cv2.rectangle(self.image, (x,y), (x+w, y+h), (255,0,0), 2)

    def draw_eye_boundaries(self):
        if self.eyes_found == False:
            return
        x1 = self.left_center[0] - self.radius
        y1 = self.left_center[1] - self.radius
        x2 = self.left_center[0] + self.radius
        y2 = self.left_center[1] + self.radius
        cv2.rectangle(self.roi, (x1,y1), (x2, y2), (0,255,0), 2)
        x1 = self.right_center[0] - self.radius
        y1 = self.right_center[1] - self.radius
        x2 = self.right_center[0] + self.radius
        y2 = self.right_center[1] + self.radius
        cv2.rectangle(self.roi, (x1,y1), (x2, y2), (0,255,0), 2)
