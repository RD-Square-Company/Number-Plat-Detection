import cv2
import numpy as np
import imutils

class NumberPlateDetector:
    'NumberPlateDetector will return all the ROI for the number plate from image'
    def __init__(self, carImage):
        'NumberPlateDetector(image_with_car(np.array)) to detect ROI for number plate'
        self.__carImage = carImage # Saving car image
        self.__frame = carImage.copy() # copying image for fasten the process for image processing
        self.__roiList = [] # empty list of roi for number plate in car image
        self.__boxes = [] # box for detected number plate rois

        # Resizing the frame for further processing
        self.__imageWidth = 300 # Setting image width to 300
        self.aspectRat = self.__imageWidth / float(self.__frame.shape[0])
        self.__imageHeight = int(self.aspectRat * self.__frame.shape[1]) # Setting image height
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

            # Making box for real image with aspect ratio
            anotherBox = []
            for row in range(box.shape[0]):
                temp = []
                for col in range(box.shape[1]):
                    temp.append(int(box[row,col] * ar))
                anotherBox.append(temp)
            self.__boxes.append(np.array(anotherBox)) # Appending the box for real image in list
        
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
        return (self.__boxes, self.__roiList)
    # getROI() ENDS
