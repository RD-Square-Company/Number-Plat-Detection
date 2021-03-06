import cv2
import numpy as np
import re
import pandas as pd
import imutils
from sklearn.svm import LinearSVC
from .pytesseract import image_to_string





video = cv2.VideoCapture('../jklu_parking_data/numberplate/a5.mp4') # Loading video in script

while True:
    # Capturing image of car
    ret, frame = video.read()
    
    if frame is None:
        break
    real = frame.copy()

    # Resizing image for number plate searching
    frame = cv2.resize(frame, (300,300))
    edges = cv2.Canny(frame, 300,400,apertureSize=3,L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    xf = real.shape[0]/frame.shape[0]
    yf = real.shape[1]/frame.shape[1]
    
    peri = cv2.arcLength(cnts[5], True)
    approx = cv2.approxPolyDP(cnts[5], 0.04*peri, True)
    (x,y, w, h) = cv2.boundingRect(approx)
    x = int(x*yf)
    y= int(y*xf)
    w = int(w*yf)
    h = int(h*xf)

    # Extracting image plate from real car image
    number_plate = real[y:y+h, x:x+w]
    number_plate = cv2.resize(number_plate, (500,100))
    n = number_plate.copy()
    number_plate = cv2.GaussianBlur(number_plate,(31,31),0)
    np_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
    np_gray = cv2.adaptiveThreshold(np_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,21,2)
    
    #cv2.imshow("gray number plate threshold", np_gray)
    cnts = cv2.findContours(np_gray.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("word with tessaract")
    print (image_to_string(n, lang='eng'))

    # Extracting letters and digits from number plate
    for c in cnts:
        if cv2.contourArea(c) < 100 or cv2.contourArea(c) > 2000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        (x,y, w, h) = cv2.boundingRect(approx)
        digit = n.copy()[y:y+h, x:x+w]
        cv2.rectangle(n, (x,y), (x+w, y+h), (0,255,0),1)
        digit = cv2.resize(digit, (4,4))
        
        cv2.imshow('Number Plate Contours', n)
        cv2.imshow("digit on number plate", digit)
        
        k = cv2.waitKey(0)
    
    k = cv2.waitKey(0)
    
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()


