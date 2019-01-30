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

    # Applying Sobel and Laplacian gradient on it

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)

    Sobel =  cv2.Sobel(frame, cv2.CV_64F,1,1,ksize=11)

    cv2.imshow('Video', edges)

    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Running")
    print("Total lengths ", len(cnts))
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        M = cv2.moments(c) # center of contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04*peri, True)
        if len(approx) == 4:
            (x,y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "rectangle"
            c = c.astype("int")
            # cv2.drawContours(frame, [c], -1, (0,255,0),1)
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),1)
            cv2.imshow(str(x), frame[y:y+h,x:x+w])
    cv2.imshow("edges", edges)
    cv2.imshow("real Image", frame)
    k = cv2.waitKey(0)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
