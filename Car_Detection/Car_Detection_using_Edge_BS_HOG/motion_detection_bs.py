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
    #frame = cv2.GaussianBlur(frame, (5,5),0)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Edge detection using canny edge detection method
    edges = cv2.Canny(frame, 100,200)

        
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = edges
        continue

    maskImage = cv2.bitwise_and(firstFrame, edges)
    maskImage1 = cv2.threshold(maskImage, 253, 255, cv2.THRESH_BINARY_INV)[1]
    #maskImage2 = cv2.threshold(maskImage, 250, 255, cv2.THRESH_BINARY)[1]
    #minus = abs(edges - maskImage2).astype(np.uint8)
    #minus = cv2.morphologyEx(minus, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    andImage = cv2.bitwise_and(edges, maskImage1)
    andImage = cv2.morphologyEx(andImage, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    
    if count > 10:
        firstFrame = edges
        count = 0
        continue
    else:
        count += 1

    cnts = cv2.findContours(andImage.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
    
    #cv2.imshow('minus', minus)
    cv2.imshow('andImage', andImage)
    cv2.imshow('real Image', frame)
      
    
    k = cv2.waitKey(30)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
