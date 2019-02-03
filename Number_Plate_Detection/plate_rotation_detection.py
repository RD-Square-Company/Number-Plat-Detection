import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

video = cv2.VideoCapture('a5.mp4') # Loading video in script

while True:
    # Capturing image of car
    ret, frame = video.read()
    
    if frame is None:
        break
    real = frame.copy() # Copying image for future use
    
    print("Image shape before resizing - ", frame.shape)
    
    # Resizing image for number plate searching while keeping aspect ration same
    imageWidth = 300 # Setting image width to 300
    aspectRat = imageWidth / float(frame.shape[0])
    imageHeight = int(aspectRat * frame.shape[1]) # Setting image height
    frame = cv2.resize(frame, (imageHeight,imageWidth)) # Resizing the image 300 * aspect Ratio height

    print("Image shape after resizing - ", frame.shape)

    # Using Canny Edge method for detecting edges in image for number plate
    edges = cv2.Canny(frame, 300,400,apertureSize=3,L2gradient=True)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    cv2.imshow("Image containing only edges", edges)

    # Calculating & Processing the number plate ROI from image
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    print("total length of cnts", len(cnts))

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

        # Making box for real image with aspect ratio
        anotherBox = []
        for row in range(box.shape[0]):
            temp = []
            for col in range(box.shape[1]):
                temp.append(int(box[row,col] * ar))
            anotherBox.append(temp)

        anotherBox = np.array(anotherBox)

        cv2.drawContours(real, [anotherBox], 0, (0,0,255),1)

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

        cv2.imshow("ROI - "+str(count),numberPlateROI) # showing the ROI for real image
        count += 1

        cv2.waitKey(0)

    # Loop breaks here for contours or ROI

    cv2.imshow("Detected ROI for Number Plate", frame)
    cv2.imshow("Detected ROI for Number Plate on Real Image", real)
    
    k = cv2.waitKey(0)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
