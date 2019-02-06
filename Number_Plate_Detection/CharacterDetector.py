import cv2
import numpy as np
import imutils
from skimage import measure
from pytesseract import image_to_string

class CharacterDetector:
    'CharacterDetector will detect characters and return the bounded rectangles for characters in number plate'

    def __init__(self, number_plate):
        'CharacterDetector(number_plate_image) will detect character and make bounded rectangles for characters in number plate'
        # Resizing the number plate roi to 400 * 100 for better character recognition
        self.__numberPlateROI = cv2.resize(number_plate, (400,100))

        # List for storing character bounding rectangle
        self.__rectList = []

        # Making copy of number plate for further use
        number_plate = self.__numberPlateROI.copy()

        # Converting the image into gray scale image for better processing of image
        np_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)

        # Thresholding the gray scale image with block size of 25 and constant 2
        # using adaptive gaussian method
        np_gray = cv2.adaptiveThreshold(np_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25,2)

        # image containing threshold image to return
        self.__threshold = np_gray.copy()

        # Image for storing hull of character for calculating bounding rectangle of characters
        hullImage = np.zeros(np_gray.shape, dtype="uint8")

        # Finding contours for the characters in the number plate
        # using LIST method instead of EXTERNAL to get all character inside
        # number plate region or ROI
        cnts = cv2.findContours(np_gray.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        # Getting contours from cnts by separating herarchies from it
        cnts = imutils.grab_contours(cnts)

        # Extracting character hull from number plate
        for c in cnts:
            # Calculating bounded rect for characters
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            # Checking for the solidity (extent) to remove additional
            # object like (logs, bolts etc.)
            extent = cv2.contourArea(c) / float(boxW * boxH)
            extent = extent > 0.2 # Accept if extent is greater than threshold

            # Checking for the aspect ratio of the characters.
            # Width of character never be greater than its height
            # so, width always be less than height or atmost equal
            # After some trials, we found that some letter like C & D and
            # due to resizing and camera angle, height of characters can be
            # lesser than its width, so we took aspect ratio less than 2 and
            # greater than 0.25

            aspectRatio = boxW / float(boxH)
            aspectRatio = (aspectRatio < 2) and (aspectRatio > 0.25) # Accept if aspect Ratio is in range

            if not (extent and aspectRatio): # Checking condition of acceptance
                continue

            # Checking for the height ratio of the characters
            # Characters on number plate can be approx equeal to the height
            # of number plate and always greater than atleast 30% of height of number plate
            # All the values are taken by giving some trials
            heightRatio = boxH / float(np_gray.shape[0])
            if not ((heightRatio > 0.3) and (heightRatio < 0.90)): # Checking conditions for acceptance
                continue

            # Calculating the hull of contour for character extraction
            hull = cv2.convexHull(c)
            cv2.drawContours(hullImage, [hull], -1, 255, -1) # drawing hull area on image

        # We are going to extract character using hull area to avoid
        # herarchy in contours
        # Extracting characters from the number plate using character hull
        # Finding contours on hull image using RETR_EXTERNAL method
        cnts = cv2.findContours(hullImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Getting contours from cnts by separting herarchies from it
        cnts = imutils.grab_contours(cnts)

        # Extracting characters from the number plate
        # with lopping on the contours of hull image
        for c in cnts:
            # Calculating and appending bounded rect for character hull to rectList
            self.__rectList.append(cv2.boundingRect(c))

        # Setting mask of hull image
        self.__mask = hullImage
    # END OF __init__()

    def getChars(self):
        'getChars() returns resized image and list of bounded rectangles of characters in number plate image'
        # Obtaining a image containing characters using
        # bitwise and of the threshold image and hull area image
        # value will be white if it is white in both the sections
        
        newImage = cv2.bitwise_and(self.__threshold, self.__mask)
        config = ("-l eng --oem 1 --psm 7")
        return image_to_string(newImage, config=config)

        
