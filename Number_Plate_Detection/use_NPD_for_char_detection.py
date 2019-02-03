import cv2
import numpy as np
import imutils
from skimage import measure

class NumberPlateDetector:
    'NumberPlateDetector will return all the ROI for the number plate from image'
    def __init__(self, carImage):
        'NumberPlateDetector(image_with_car(np.array)) to detect ROI for number plate'
        self.__carImage = carImage # Saving car image
        self.__frame = carImage.copy() # copying image for fasten the process for image processing
        self.__roiList = [] # empty list of roi for number plate in car image

        # Resizing the frame for further processing
        self.__imageWidth = 300 # Setting image width to 300
        self.aspectRat = self.__imageWidth / float(frame.shape[0])
        self.__imageHeight = int(self.aspectRat * frame.shape[1]) # Setting image height
        self.__frame = cv2.resize(self.__frame, (self.__imageHeight, self.__imageWidth)) # Resizing the image 300 * aspect Ratio height of image

        # Setting extent, aspectRatio of number_plate, heightRatio
        self.extent = 0.3 # default setting
        self.aspectRatio = 1.2 # default aspect ratio for number plate roi
        self.heightRatio = (7, 17) # default tuple to store lower and upper bound of height ratio for roi
        
    # __init__() IS END...

    def __findROI(self):
        'Find the roi regions of number plate in car image'
        # Setting list of roi's to the empty
        self.__roiList = []
        
        # Using Canny Edge method for detecting edges in car image for number plate
        # lowerBound is 300 and upper bound is 400
        edges = cv2.Canny(self.__frame, 300, 400, apertureSize=3, L2gradient=True)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))) # Dilating edges for better detection of number plate roi's

        # Calculating & Processing the number plate ROI from image
        cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts) # getting contours by separating it from herarchies

        # Extracting the number plate ROI from car image
        for c in cnts:
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c) # finding bounding rect straight for ROI

            # Checking for the solidity (extent) of the contour to avoid
            # unacceptable items like nut-bolts, logos etc.
            extent = cv2.contourArea(c) / float(boxW * boxH)
            extent = extent > self.extent # Accept only if solidity (extent) is greater than threshold value

            # Checking for the aspect ratio of the number plate roi
            # number plate's width is atleast more than twice of its height
            # but due to the rotation or due to camera angle it will may become lesser
            # than twice of its height, so we assuming it always be greater than height
            aspectRatio = boxW / float(boxH)
            aspectRatio = aspectRatio > self.aspectRatio # Accept only if width is more than height for roi

            # Continue if solidity (extent) and aspect ratio is not in range
            if not (extent and aspectRatio):
                continue

            # If we devide height of car by height of number plate then
            # the ratio comes in between 7 to 17,
            # So by keeping this in min (temporary, if not work for other data - will be removed)
            # ROI with lower than 7 and greater than 17 will be removed
            heightRatio = self.__imageHeight / boxH # Calculating height ratio for ROI

            if (heightRatio < self.heightRatio[0] or heightRatio > self.heightRatio[1]): # Checking for height ratio of roi
                continue

            rect  = cv2.minAreaRect(c) # Finding minimum area rectangle
            box = cv2.boxPoints(rect) # Finding rotated box's points or hull set for rectangle
            box = np.int0(box)

            # Separating rotated rectangle from the image
            ar = self.__carImage.shape[0] / self.__frame.shape[0] # Setting aspect ratio for image

            # Getting height and width of rotated or minimum area rectangle
            W = int(rect[1][0] * ar)
            H = int(rect[1][1] * ar)

            # Getting all x and y saved points of rotated rect (hull set)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]

            # Finding min, max of X's and Y's points of rotated rect
            x1, x2 = int(min(Xs)*ar), int(max(Xs)*ar)
            y1, y2 = int(min(Ys)*ar), int(max(Ys)*ar)

            # Let, rotated rectangle is not inverted
            rotated = False
            angle = rect[2] # Getting angle of rotation of rectangle

            if angle < -45: # Operation for switching between width and height of rectangle
                angle += 90
                rotated = True

            center = (int((x1+x2)/2), int((y1+y2)/2)) # Center of ROI for real image
            size = (int(x2-x1), int(y2-y1)) # size of ROI for real image

            # Getting rotation matrix of rotated rectangle or ROI
            rotationMatrix = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

            # clipping bounded rectangle of ROI for real image
            boundedROI = cv2.getRectSubPix(self.__carImage, size, center)
            transformROI = cv2.warpAffine(boundedROI, rotationMatrix, size) # Rotating ROI with rotation matrix for real image

            # Switching between height and width of ROI if it is inverted
            croppedW = W if not rotated else H
            croppedH = H if not rotated else W

            # Getting only Rotated ROI from transformed ROI for real image
            numberPlateROI = cv2.getRectSubPix(transformROI, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))

            # Appending ROIs to the self.__roiList
            self.__roiList.append(numberPlateROI)
            
        # For LOOP ENDS
    # __findROI() ENDS

    def get_roi(self):
        'get_roi() will return list of roi\'s for number plate in car image'
        self.__findROI() # Finding the ROI for number plate in car image
        return self.__roiList
    # getROI() ENDS




####################################TESTING OUR NUMBERPLATEDETECTOR FOR IMAGES########################
#../jklu_parking_data/numberplate/-
video = cv2.VideoCapture('0.mp4') # Loading video in script

while True:
    # Capturing image of car
    ret, frame = video.read()
    
    if frame is None:
        break
    real = frame.copy() # Copying image for future use

    plateDetector = NumberPlateDetector(frame)
    plates = plateDetector.get_roi()

    for plate in plates:
        # Resizing the number plate roi to 500 * 100 for better character recognition
        numberPlateROI = cv2.resize(plate, (400,100))
        cv2.imshow("ROI", numberPlateROI)
        number_plate = numberPlateROI.copy()

        # Convertig the image into gray scale image for better processing of image
        np_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)

        # Thresholding the gray scale image with 21 x 21 kernel
        # and using adaptive threshold method
        np_gray = cv2.adaptiveThreshold(np_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25,2)
        cv2.imshow("threshold gray", np_gray)

        # image for hull of characters
        hullImage = np.zeros(np_gray.shape, dtype="uint8")
        
        # Finding countours for the characters in the number plate
        # using LIST method instead of EXTERNAL to get all character inside
        # number plate region or ROI
        cnts = cv2.findContours(np_gray.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)

        # Getting contours from cnts by separating herarchies from it
        cnts = imutils.grab_contours(cnts)

        # Extracting character hull from number plate
        for c in cnts:
            # Calculating bounded rect for characters
            (x, y, w, h) = cv2.boundingRect(c)

            # Checking for the solidity (extent) to remove additional
            # object (like logos, bolts etc.)
            solidity = cv2.contourArea(c) / float(w*h)
            solidity = solidity > 0.2 # Accept if solidity is greater than threshold

            # Checking for the aspect Ratio of the characters
            # Width of character never be greater than its height
            # so, check for width always be less than height or atmost equal
            # After some trials, we found that some letter like C & D have width equal to
            # height but due to camera angle its height can be small lesser than its width
            # Hance, we are taking aspect ratio of 1.2 for compact texts on number plate
            aspectRatio = w / float(h)
            aspectRatio = (aspectRatio < 2) and (aspectRatio > 0.25) # Accept if aspect Ratio is less than 1.2

            if not (solidity and aspectRatio): # Checking condition for solidity
                continue

            # Checking for the height ration of the characters
            # Characters on number plate can be approx equal to the height
            # of number plate or either greater than atleast 30% height of number plate
            # for two lines on number plate
            heightRatio = h / float(np_gray.shape[0])
            if not ((heightRatio > 0.3) and (heightRatio < 0.90)):
                continue

            # Calculating the hull of contour for character extraction
            hull = cv2.convexHull(c)
            cv2.drawContours(hullImage, [hull], -1,255, -1)


        # Extracting characters from number plate using character hull
        # Finding contours on hull image using RETR_EXTERNAL method
        cnts = cv2.findContours(hullImage.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        # Getting contours from cnts by separating herarchies from it
        cnts = imutils.grab_contours(cnts)

        # Extracting characters from the number plate
        # with looping on the contours of hull image
        for c in cnts:
            # Calculating bounded rect for character hull
            (x, y, w, h) = cv2.boundingRect(c)

            cv2.rectangle(numberPlateROI, (x,y), (x+w, y+h) , (0,255,0))
        
            
        cv2.imshow("number plate contours", numberPlateROI)
        cv2.imshow("Hull image", hullImage)
        cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
