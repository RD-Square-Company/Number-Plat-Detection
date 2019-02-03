import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from skimage import measure

video = cv2.VideoCapture('a6.mp4') # Loading video in script

while True:
    # Capturing image of car
    ret, frame = video.read()
    
    if frame is None:
        break
    real = frame.copy() # Copying image for future use
    
    # Resizing image for number plate searching while keeping aspect ration same
    imageWidth = 300 # Setting image width to 300
    aspectRat = imageWidth / float(frame.shape[0])
    imageHeight = int(aspectRat * frame.shape[1]) # Setting image height
    frame = cv2.resize(frame, (imageHeight,imageWidth)) # Resizing the image 300 * aspect Ratio height

    # Using Canny Edge method for detecting edges in image for number plate
    edges = cv2.Canny(frame, 300,400,apertureSize=3,L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    #cv2.imshow("Image containing only edges", edges)

    # Calculating & Processing the number plate ROI from image
    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Extracting the number plate ROI from image
    count = 0 # number of setting roi in cv2.imshow()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c) # finding bounding rect straight

        # Checking for the solidity (extent) of the contour to avoid
        # unacceptable items like nut-bolts, logos etc.
        extent = cv2.contourArea(c) / float(w*h)
        extent = extent > 0.3 # Accept only if solidity is greater than threshold value

        # Checking for the aspect ration of the number plate
        # number plate's width is atleast more than twice of its height
        # but due to the rotation in or due to camera angle it will may become lesser
        # than twice of its height, so we assuming it always be greater than height
        aspectRatio = w / float(h)
        aspectRatio = aspectRatio > 1.2 # Accept only if width is more than height

        # Continue if solidity and aspect ration is in range
        if not (extent and aspectRatio) :
            continue

        # If we devide height of car by height of number plate then
        # the ratio comes in between 7 to 17,
        # So by keeping this in mind (temporary, if not work for other data - will be removed)
        # ROI with lower than 7 and greater than 17 will be removed
        heightRatio = imageHeight / h # Calculating height ratio for ROI
    
        if (heightRatio < 7) or (heightRatio > 17):
            continue
        
        rect = cv2.minAreaRect(c) # Finding minimum area rectangle
        box = cv2.boxPoints(rect) # Finding rotated box for rectangle
        box = np.int0(box)

        #sRect = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1)
        rRect = cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)

        # Separating Rotated rectangle from the image
        ar = real.shape[0] / frame.shape[0] # Setting aspect ratio for image

        # Getting height and width of rotated or minimum area rectangle
        W = int(rect[1][0] * ar) 
        H = int(rect[1][1] * ar)

        # Getting all x and y saved points of rotated rect
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]

        # Finding min, max of x's and y's points of rotated rect
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
        #cv2.circle(real, center, 5, (0,255,0), -1) # for debug only

        # Getting rotation matrix of rotated rectangle or ROI
        rotationMatrix = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

        # clipping bounded rectangle of ROI for real image
        boundedROI = cv2.getRectSubPix(real, size, center)
        transformROI = cv2.warpAffine(boundedROI, rotationMatrix, size) # Rotating ROI with rotation matrix for real image
        

        # Switching between height and width of ROI if it is inverted
        croppedW = W if not rotated else H
        croppedH = H if not rotated else W

        # Getting only Rotated ROI from transformed ROI for real image
        numberPlateROI = cv2.getRectSubPix(transformROI, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))

        #cv2.imshow("ROI - "+str(count),numberPlateROI) # showing the ROI for real image
        count += 1

        '''###################################################################################
        ###################################################################################
        ###################################################################################'''

        # Resizing the number plate roi to 500 * 100 for better character recognition
        numberPlateROI = cv2.resize(numberPlateROI, (400,50))
        cv2.imshow("ROI", numberPlateROI)
        number_plate = numberPlateROI.copy()

        # Blurring the ROI image with () kernel and gaussian blur method
        # for reducing the noise and sharp edges
        #number_plate = cv2.GaussianBlur(number_plate, (7,7),0)
        cv2.imshow("blur roi", number_plate)

        # Convertig the image into gray scale image for better processing of image
        np_gray = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)

        # Thresholding the gray scale image with 21 x 21 kernel
        # and using adaptive threshold method
        np_gray = cv2.adaptiveThreshold(np_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25,2)
        cv2.imshow("threshold gray", np_gray)


        # Connected Components Analysis for separating characters from the number
        # plate of car
        # Calculating labels for pixels in image
        imageLabels = measure.label(np_gray, neighbors=8, background=0)

        # Loop over the each label and putting all into image mask of characters
        for label in np.unique(imageLabels):
            # if label is background label than skip it
            if label == 0:
                continue

            # Otherwise, put that pixel into imageMask for further processing
            # Creating a mask of the characters in the number plate
            imageMask = np.zeros(np_gray.shape, dtype="uint8")
            imageMask[imageLabels == label] = 255

            #cv2.imshow("threshold after CCA", imageMask)
            #cv2.waitKey(0)

        # Finding countours for the characters in the number plate
        # using LIST method instead of EXTERNAL to get all character inside
        # number plate region or ROI
        cnts1 = cv2.findContours(np_gray.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)

        # Getting contours from cnts by separating herarchies from it
        cnts1 = imutils.grab_contours(cnts1)

        # Extracting characters from number plate
        for c1 in cnts1:
            # Calculating bounded rect for characters
            (x, y, w, h) = cv2.boundingRect(c1)

            # Checking for the solidity (extent) to remove additional
            # object (like logos, bolts etc.)
            solidity = cv2.contourArea(c1) / float(w*h)
            solidity = solidity > 0.3 # Accept if solidity is greater than threshold

            # Checking for the aspect Ratio of the characters
            # Width of character never be greater than its height
            # so, check for width always be less than height or atmost equal
            # After some trials, we found that some letter like C & D have width equal to
            # height but due to camera angle its height can be small lesser than its width
            # Hance, we are taking aspect ratio of 1.2 for compact texts on number plate
            aspectRatio = w / float(h)
            aspectRatio = aspectRatio < 1.2 # Accept if aspect Ratio is less than 1.2

            #if not ( solidity and aspectRatio): # Checking condition for solidity
                #continue

            # Checking for the height ration of the characters
            # Characters on number plate can be approx equal to the height
            # of number plate or either greater than atleast 30% height of number plate
            # for two lines on number plate
            heightRatio = h / float(np_gray.shape[0])
            #if not ((heightRatio > 0.3) and (heightRatio < 0.90)):
                #continue
            
            cv2.rectangle(numberPlateROI, (x,y),(x+w, y+h),(0,255,0),1)
            cv2.imshow("number plate contours", numberPlateROI)
        
        cv2.waitKey(0)
        

    # Loop breaks here for contours or ROI

    cv2.imshow("Detected ROI for Number Plate", frame)
    #cv2.imshow("Detected ROI for Number Plate on Real Image", real)
    
    k = cv2.waitKey(0)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
