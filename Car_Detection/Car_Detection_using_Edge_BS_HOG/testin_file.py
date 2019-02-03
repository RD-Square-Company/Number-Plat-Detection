import numpy as np
import cv2

cap = cv2.VideoCapture(0)



while(1):
    if (cap.isOpened() == False):
        print("Error opening Video file")
        break

    ret,frame = cap.read()
    if frame is None:
        break

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
