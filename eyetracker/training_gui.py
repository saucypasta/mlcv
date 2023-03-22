from tkinter import *
import VideoCapture
from csv import writer
import numpy as np
import cv2
import EyeTracking


class TrainingApp:
    def __init__(self, root, cols = 6, rows = 4, radius = 50, video_source = 0, file_name = "training_data.csv"):
        self.root = root
        self.cols = cols
        self.rows = rows
        self.circle_radius = radius
        self.video_source = video_source
        self.vid = VideoCapture.MyVideoCapture(self.video_source)
        self.eye_tracker = EyeTracking.EyeTracker(self.vid)
        self.calib_rdy = 0
        self.calibrated = False
        cv2.namedWindow('image')
        cv2.createTrackbar('threshold', 'image', 42, 255, self.set_thresh)
        self.left_eye = []
        self.file_name = file_name
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.circles = []
        self.left = []
        self.right = []
        self.color_ind = 2
        self.clicked = "cyan"
        self.eye_detected = "green"
        self.no_detection = "red"
        self.data = []
        self.col_scale = (self.screen_width - (self.circle_radius * 2))/self.cols
        self.row_scale = (self.screen_height - (self.circle_radius * 2))/self.rows
        self.pcol_scale = 0
        self.pcol_offset = 0
        self.prow_scale = 0
        self.prow_offset = 0
        #center, top, left, right, bottom
        self.coords = [0,0,0,0,0]
        self.canvas = Canvas(self.root, width = self.screen_width, height=self.screen_height, bg="white")
        self.canvas.bind("<Button-1>", self.mouse_pressed)
        self.init_circles()
        self.canvas.pack()
        self.delay = 1
        self.update()
        self.root.mainloop()

    def set_thresh(self,val):
        pass

    def draw_circle(self, circle, radius=50):
        (center_x, center_y, color) = circle
        self.canvas.create_oval(center_x-radius, center_y - radius,
                           center_x+radius, center_y + radius, fill=color)

    def find_nearest_circle(self, x, y):
        col = round(x / self.col_scale)
        row = round(y / self.row_scale)
        ind = row * (self.cols+1) + col
        return ind

    def calibrate(self):
        # center = 17
        # top = 3
        # left = 14
        # right = 20
        # bottom = 31
        (centerx, centery, color) = self.circles[17]
        (topx, topy, color) = self.circles[3]
        (leftx, lefty, color) = self.circles[14]
        (rightx, righty, color) = self.circles[20]
        (bottomx, bottomy, color) = self.circles[31]

        (cx,cy) = self.coords[0]
        (tx,ty) = self.coords[1]
        (lx,ly) = self.coords[2]
        (rx,ry) = self.coords[3]
        (bx,by) = self.coords[4]

        print("cx, cy: ", cx,cy)
        print("tx, ty: ", tx,ty)
        print("lx, ly: ", lx,ly)
        print("rx, ry: ", rx,ry)
        print("bx, by: ", bx,by)
        self.pcol_scale = (rightx - leftx)/(rx - lx)
        self.pcol_offset = lx - leftx
        self.prow_scale = (bottomy-topy)/(ty - by)
        self.prow_offset = ty

        print("c_scale ", self.pcol_scale)
        print("c_offset", self.pcol_offset)
        print("r_scale ", self.prow_scale)
        print("r_offset", self.prow_offset)

        self.calibrated = True

    #center x, center y, left eye x, left eye y, right eye x, right eye y
    def append_training(self, ind, data):
        (center_x, center_y, color) = self.circles[ind]
        row = [center_x/self.screen_width, center_y/self.screen_height]
        [centers, ears, face_points] = data
        avex = 0
        avey = 0
        for i in range(0,len(centers)):
            (x,y) = centers[i]
            avex = avex + x
            row.append(x/1280)
            row.append(y/720)
        avex = avex/2
        for ear in ears:
            avey = avey + ear
            row.append(ear)
        avey = avey/2
        # center = 17
        # top = 3
        # left = 14
        # right = 20
        # bottom = 31
        if(self.calibrated):
            x = (avex - self.pcol_offset) * self.pcol_scale
            y = (self.prow_offset - avey) * self.prow_scale
            print("curx: ",x)
            print("cury: ",y)
        if(ind == 17):
            self.coords[0] = [avex,avey]
            self.calib_rdy = self.calib_rdy | 1
            print("center")
        if(ind == 3):
            self.coords[1] = [avex,avey]
            self.calib_rdy = self.calib_rdy | 2
            print("top")
        if(ind == 14):
            self.coords[2] = [avex,avey]
            self.calib_rdy = self.calib_rdy | 4
            print("left")
        if(ind == 20):
            self.coords[3] = [avex,avey]
            self.calib_rdy = self.calib_rdy | 8
            print("right")
        if(ind == 31):
            self.coords[4] = [avex,avey]
            self.calib_rdy = self.calib_rdy | 16
            print("bottom")
        if(self.calibrated == False and self.calib_rdy == 0b11111):
            self.calibrate()
        for (x,y) in face_points:
            row.append(x/1280)
            row.append(y/720)
        with open(self.file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(row)

    def mouse_pressed(self,event):
        data = self.data
        if(data != []):
            ind = self.find_nearest_circle(event.x, event.y)
            self.circles[ind][self.color_ind] = self.clicked
            self.append_training(ind, data[1])

    def init_circles(self):
        for i in range(0, self.rows+1):
            for j in range(0, self.cols+1):
                center_x = j*self.col_scale + self.circle_radius
                center_y = i*self.row_scale + self.circle_radius
                self.circles.append([center_x, center_y, self.eye_detected])
        for circle in self.circles:
            self.draw_circle(circle)


    # def eye_update(self):
    #     self.data = self.eye_tracker.mainloop()
    #     self.root.after(1,self.eye_update())

    def update(self):
        thresh_val = cv2.getTrackbarPos('threshold', 'image')
        self.eye_tracker.pupil_thresh = thresh_val
        self.data = self.eye_tracker.mainloop()
        if(self.data != []):
            cv2.imshow('image', self.data[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
        self.root.after(self.delay, self.update)




root = Tk()
root.state("zoomed")
TrainingApp(root)
cv2.destroyAllWindows()
