import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture('camera.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = np.ones((5,5),np.uint8)
while(1):
    ret,frame = cap.read()
    if frame is None:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    fgmask = fgbg.apply(frame)
    image = cv2.dilate(fgmask, kernel, iterations=2)

    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) < 1000:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
    
    cv2.imshow('frame', image)
    cv2.imshow('real detection', frame)
    
    k = cv2.waitKey(30)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
