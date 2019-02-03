import cv2
import pytesseract
from NumberPlateDetector import *
from CharacterDetector import *


#../jklu_parking_data/numberplate/-
video = cv2.VideoCapture('a5.mp4') # Loading video in script
count = 0
while True:
    # Capturing image of car
    ret, frame = video.read()
    if count == 1:
        count = 0
        continue
    count += 1
    if frame is None:
        break
    real = frame.copy() # Copying image for future use
    
    plateDetector = NumberPlateDetector(frame) # Number plate detector
    boxes, plates = plateDetector.get_roi() # getting number plates from detector
    
    for box, plate in zip(boxes, plates):
        #cv2.imshow("plate", plate)
        #cv2.drawContours(real, [box], 0, (255,0,255),3) # drawing box
        charDetector = CharacterDetector(plate) # Character detector
        text = charDetector.getChars()
        print(text)

    cv2.imshow("Real Image", real)
    k = cv2.waitKey(30)
    if k == 27:
        break
    #cv2.imwrite("car_number_plate.png", real)
    #cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
