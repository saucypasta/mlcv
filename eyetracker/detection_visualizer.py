import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import Finder
import VideoCapture

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

         # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture.MyVideoCapture(self.video_source)
         # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.finder = Finder.FeatureFinder()
        self.update()


        self.window.mainloop()


    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
             self.finder.set_image(frame)
             self.finder.find_face()
             self.finder.find_eyes()
             self.finder.draw_face_boundary()
             self.finder.draw_eye_boundaries()
             self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.finder.get_image()))

             self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
         if(self.finder.threshold != []):
             cv2.imshow('image', self.finder.threshold)
             cv2.waitKey(10)
         self.window.after(self.delay, self.update)


 # Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
