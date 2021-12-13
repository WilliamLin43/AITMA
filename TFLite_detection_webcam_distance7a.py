#version 7
#reads from config.txt file
#added move foward option- this will move foward slightly
#   just in case there is a false positive that lasts a long time
#
# Import packages
import os
#import argparse
import configparser
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from os import listdir
from os.path import isfile,join

import RPi.GPIO as gpio
gpio.setmode(gpio.BCM)
gpio.setup(14,gpio.OUT)

#from inArea import inArea
from beeper import beeper
#beep=beeper()
import utils
def listfiles(path,extension):
    return [f for f in listdir(path) if (isfile(join(path,f)) and f.endswith(str(extension)))]

from VideoStream import VideoStream, SaveFrame


# Define and parse input arguments
config=configparser.ConfigParser()
config.read("config.txt")

'''
for k in config["DEFAULT"]:
    print(str(k)+" = "+str(config["DEFAULT"][k]))
#time.sleep(5)
'''
imagepath=str(config["DEFAULT"]["imagepath"])
imagepathnobox=str(config["DEFAULT"]["imagepathnobox"])
numImages=len(listfiles(imagepath,".png"))

MODEL_NAME = str(config["DEFAULT"]["modeldir"])#args.modeldir
GRAPH_NAME = str(config["DEFAULT"]["graph"])#args.graph
LABELMAP_NAME = str(config["DEFAULT"]["labels"])#args.labels
min_conf_threshold = float((config["DEFAULT"]["threshold"]))#float(args.threshold)
resW, resH = str(config["DEFAULT"]["resolution"]).split('x')#args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = True if config["DEFAULT"]["edgetpu"]=="True" else False#bool(config["DEFAULT"]["edgetpu"])#args.edgetpu

distance = float(config["DEFAULT"]["distance"])#float(args.distance)
minsize = float(config["DEFAULT"]["minsize"])
distanceCurve=config["DEFAULT"]["distancecurve"]

PersonDetectThreshold=int(config["DEFAULT"]["persondetectthreshold"])
NotPersonDetectThreshold=int(config["DEFAULT"]["notpersondetectthreshold"])
AGVPauseTime=int(config["DEFAULT"]["pausetime"])

MoveForwardFailSafe=True if config["DEFAULT"]["moveforwardfailsafe"]=="True" else False
MoveForwardFailSafeTime=float(config["DEFAULT"]["moveforwardtime"])
FPWaitTime=float(config["DEFAULT"]["fpwaittime"])

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate("libedgetpu.so.1")])
    #interpreter = Interpreter(model_path=PATH_TO_CKPT,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print("model dimensions: (" + str(width)+","+str(height)+")")
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=60).start()
saveframe = SaveFrame(imagepath,imagepathnobox,numImages).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
timesaved=time.time()
timeThresh=5
detectPerson=False
lPoints=[(0,imH),(int(imW/2-imW*.05),int(imH*.1))]
rPoints=[(imW,imH),(int(imW/2+imW*.05),int(imH*.1))]
#for making sure no false detection
detectPersonThresh=0
NodetectPersonThresh=0
#for sounds
prevDetectPerson=False
prevDetectPersonlist=[False]*NotPersonDetectThreshold
soundCounter=time.time()

#restart
startRestart=True
restartTime=time.time()-4
AGVON=False
AGVBEEPING=False
detectPersonTooClose=False

#FP go forward
FPMoved=False
FPMoving=False
while True:
    config.read("/home/pi/Desktop/wifiscan/AGVStatus.txt")
    AGV_Status=int(config["DEFAULT"]["AGV_Status"])
    
    prevDetectPerson=detectPersonTooClose#detectPerson
    prevDetectPersonlist.pop()
    prevDetectPersonlist.insert(0,detectPersonTooClose)
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    frame1 = cv2.flip(frame1,0)
    
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    framesave=frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    '''
    #draw the lines for a guide
    frame=cv2.line(frame,lPoints[0],lPoints[1],(0,0,255),1)
    frame=cv2.line(frame,rPoints[0],rPoints[1],(0,0,255),1)
    '''
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    
    temp=False
    detectPersonTooClose=False
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            xmid = int((xmax+xmin)/2)
            ymid = int((ymax+ymin)/2)
            
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'#'c:%d%% - d:%s' % (int(scores[i]*100),apx_distance) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
            #cv2.rectangle(framesave, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            #cv2.putText(framesave, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text
            
            if int(classes[i]) == 0 and ((xmax-xmin)*(ymax-ymin)/(imH*imW)>minsize):
                
                #print(detectPersonThresh)
                temp=True

                apx_distance=round((1-(xmax-xmin)*(ymax-ymin)/(imH*imW))**4,4)
                apx_distance=apx_distance*eval(distanceCurve)
                #cv2.putText(frame,str(apx_distance),(xmid, ymid), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                detectPersonThresh+=1
                #draw boxes and center of boxes
                if apx_distance<distance:
                    if True:#inArea((width, height),(xmid,ymid)):
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 10, 255), 2)
                        #cv2.rectangle(framesave, (xmin,ymin), (xmax,ymax), (0, 10, 255), 2)
                    else:
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 10, 0), 2)
                        #cv2.rectangle(framesave, (xmin,ymin), (xmax,ymax), (255, 10, 0), 2)
                else:
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    #cv2.rectangle(framesave, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                frame=cv2.circle(frame,(xmid,ymid),7,(255,0,0),2)
                #framesave=cv2.circle(framesave,(xmid,ymid),7,(255,0,0),2)
                
                #print(detectPersonThresh)
                
                if detectPersonThresh>=PersonDetectThreshold: #apx_distance<distance and
                    if True:#inArea((width, height),(xmid,ymid)):
                        detectPerson=True
                        if apx_distance<distance:
                            detectPersonTooClose=True
                        if time.time()-timesaved>timeThresh:
                            #cv2.imwrite(imagepath+"/"+str(numImages)+".png",frame)
                            #cv2.imwrite(imagepathnobox+"/"+str(numImages)+".png",framesave)
                            #numImages+=1
                            #print("image saved " +str(numImages-1))
                            saveframe.saveImage(frame,framesave)
                            timesaved=time.time()
 
    if temp==False:
        detectPersonThresh=0
        detectPerson=False
        NodetectPersonThresh+=1
        detectPersonTooClose=False
    else:
        NodetectPersonThresh=0
    if detectPersonTooClose==True and prevDetectPerson==False:
        soundCounter=time.time()
    
    if detectPersonTooClose:#detectPerson:
        if FPMoving==False and AGV_Status ==1:
            AGVON=False
        #AGV False Positive move a little bit recovery 
        if MoveForwardFailSafe:
                if time.time()-soundCounter>=FPWaitTime:
                    if FPMoved==False:
                        FPMoving=True
                        AGVON=True
                        if time.time()-soundCounter>=FPWaitTime+MoveForwardFailSafeTime:
                            #FPMoved=True
                            FPMoving=False
                            soundCounter=time.time()
                            FPMoved=False
        if time.time()-soundCounter>=5:
            #play sounds
            #print("BEEP")
            AGVBEEPING=True
            #pass
            #beep.beep()
            
            
            
                    
        #print("on")

    if detectPersonTooClose==False and prevDetectPersonlist[NotPersonDetectThreshold-1]==True:
        startRestart=True
        restartTime=time.time()
        FPMoved=False
        FPMoving=False
    if startRestart==True and not detectPersonTooClose:
        #beep.stop()
        AGVBEEPING=False
        if time.time()-restartTime>=AGVPauseTime:
            startRestart=False
            AGVON=True                
            #print("AGV ON")
            
    if AGVON:
        gpio.output(14,gpio.LOW)
    else:
        gpio.output(14,gpio.HIGH)
        #print("off")
    
    # Draw framerate in corner of frame
    #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1,cv2.LINE_AA)
    cv2.imshow("info",utils.makeinfowindow(['FPS: {0:.2f}'.format(frame_rate_calc),
                                            "pictures: "+str(numImages),
                                            "person thresh: "+str(min(detectPersonThresh,PersonDetectThreshold))+"/"+str(PersonDetectThreshold),
                                            "No person thresh: "+str(min(NodetectPersonThresh,NotPersonDetectThreshold))+"/"+str(NotPersonDetectThreshold),
                                            "restart time {0:.2f}: ".format(time.time()-restartTime),
                                            "    AGV ON: "+str(AGVON),
                                            "beeping timer {0:.2f}: ".format(time.time()-soundCounter),
                                            "    Beeping: "+str(AGVBEEPING),
                                            "AGV test FP: "+str(FPMoved),
                                            "AGV Status: "+str(AGV_Status)]))
                                            # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
saveframe.stop()
gpio.output(14,gpio.LOW)
