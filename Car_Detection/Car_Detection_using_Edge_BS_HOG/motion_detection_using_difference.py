# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

video = cv2.VideoCapture('camera.mp4') # Capturing video from camera

# Initialize the first frame in the video stream
firstFrame = None
count = 3
kernel =np.ones((5,5),np.uint8)
# loop over the frames of the video
while True:
    ret, frame = video.read()

    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21),2)

        
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
        

    # Compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray) 
    thresh = cv2.threshold(frameDelta, 10, 255,cv2.THRESH_BINARY)[1] # use inverse binary threshold when needed

    # dilate the thresholded image to fill in holes,
    # then find contours on thresholded image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)

    if count > 1:
        firstFrame = gray
        count = 0
        continue
    else:
        cv2.imshow('frame1', frame)  # Checking for the different conditions
        #cv2.imshow('threshold', thresh)
        count += 1
    
    
    k = cv2.waitKey(30)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
