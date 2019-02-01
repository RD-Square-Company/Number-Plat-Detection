import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

video = cv2.VideoCapture('jklu_parking_data/numberplate/a5.mp4') # Loading video in script

while True:
    ret, frame = video.read()
    

    if frame is None:
        break

    frame = cv2.resize(frame, (300,300))

    edges = cv2.Canny(frame, 300,400,apertureSize=3,L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    print("after grab...")
    
    #cv2.drawContours(frame, [cnts[5]], -1, (255,255,255),1)
    
    peri = cv2.arcLength(cnts[5], True)
    approx = cv2.approxPolyDP(cnts[5], 0.04*peri, True)
    (x,y, w, h) = cv2.boundingRect(approx)

    number_plate = frame[y:y+h,x:x+w]
    number_plate = cv2.resize(number_plate, (500,100))
    n = number_plate.copy()
    cv2.imshow("number plate", number_plate)
    number_plate = cv2.GaussianBlur(number_plate,(31,31),0)
    cv2.imshow("blurred Number plate", number_plate)
    np_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
    np_gray = cv2.adaptiveThreshold(np_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,21,2)

    #r, np_gray = cv2.threshold(np_gray, 170, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("gray number plate threshold", np_gray)

    ''' Starting with contours to detect digits and letters
    on number plate then, we will go for threshold of image to check for
    better performance'''
    #nEdge = cv2.Canny(number_plate, 300,400,apertureSize=3,L2gradient=True)
    #cv2.imshow('Number Plate Edge', nEdge)
    cnts = cv2.findContours(np_gray.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cv2.drawContours(number_plate, cnts, -1, (0,255,0),1)
    
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        (x,y, w, h) = cv2.boundingRect(approx)
        cv2.rectangle(n, (x,y), (x+w, y+h), (0,255,0),1)
    
        cv2.imshow('Number Plate Contours', n)
        k = cv2.waitKey(0)
    
    k = cv2.waitKey(0)
    
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
