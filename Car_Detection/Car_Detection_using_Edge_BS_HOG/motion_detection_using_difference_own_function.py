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
count = 0
kernel =np.ones((5,5),np.uint8)
# loop over the frames of the video
while True:
    ret, frame = video.read()

    if frame is None:
        break


    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    #frame = cv2.GaussianBlur(frame, (3,3),0)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = frame
        continue
        
    # Compute the absolute difference between the current frame and first frame
    diffFrame = cv2.absdiff(firstFrame, frame)
    threshold = 5 # threshold value for binarizing image
    gray = cv2.cvtColor(diffFrame, cv2.COLOR_BGR2GRAY)
    diffFrame = np.sqrt(diffFrame[:,:,0]^2 + diffFrame[:,:,1]^2 + diffFrame[:,:,2]^2).astype(np.uint8)
    diffFrame = cv2.threshold(diffFrame, threshold, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    

    #eroded = cv2.erode(diffFrame, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    cv2.imshow('Testing Frame', diffFrame)
    cv2.imshow('original frame', frame)
    cv2.imshow('gray frame difference', gray)
    
    if count > 1:
        firstFrame = frame
        count = 0
        continue
    else:
        #cv2.imshow('frame1', diffFrame)  # Checking for the different conditions
        #cv2.imshow('threshold', thresh)
        count += 1
      
    
    k = cv2.waitKey(30)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
