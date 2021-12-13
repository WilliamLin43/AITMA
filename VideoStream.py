import cv2
import numpy as np
from threading import Thread
import time
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(300,300),framerate=60):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True


class SaveFrame:
    def __init__(self,imagepath,imagepathnobox,numImages):
        self.frame=None
        self.framenobox=None
        self.save=False
        self.imagepath=imagepath
        self.imagepathnobox=imagepathnobox
        # Variable to control when the camera is stopped
        self.stopped = False
        self.numImages=numImages

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            time.sleep(.01)
            if self.save==True:
                try:
                    cv2.imwrite(self.imagepath+"/"+str(self.numImages)+".png",self.frame)
                    cv2.imwrite(self.imagepathnobox+"/"+str(self.numImages)+".png",self.framenobox)
                    self.numImages+=1
                    print("image saved " +str(self.numImages-1))
                except:
                    print("failed to save")
                self.save=False

    def saveImage(self,frame,framenobox):
        self.frame=frame
        self.framenobox=framenobox
        self.save=True

    def stop(self):
        self.stopped = True