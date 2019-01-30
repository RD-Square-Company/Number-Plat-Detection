# import the necessary packages
import imutils
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from skimage.feature import hog

#                           START READING FILES FOR VEHICLES AND NON-VEHICLES
# Reading car images
vehicle_image_arr1 = glob.glob('/media/rd_square/Important/Image_processing/car_data_with_neg/vehicles/*/*.png')

# read images, flip images and append into list
vehicle_images_list = []
for imagePath in vehicle_image_arr1:
    readImage = cv2.imread(imagePath)
    rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
    rgbImage = cv2.resize(rgbImage, (64,64))
    flippedImage = cv2.flip(rgbImage, flipCode=1) # flip horizontally
    vehicle_images_list.append(rgbImage)
    vehicle_images_list.append(flippedImage)
    
print("Reading of vehicle images done...")

# Reading non car images
no_vehicle_image_arr1 = glob.glob('/media/rd_square/Important/Image_processing/car_data_with_neg/non-vehicles/*/*.png')
no_vehicle_image_arr2 = glob.glob('/media/rd_square/Important/Image_processing/haar_cascade_classifier/neg/*.jpg')

# read images, flip images and append into list
no_vehicle_images_list = []
for imagePath in no_vehicle_image_arr1:
    readImage = cv2.imread(imagePath)
    rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
    rgbImage = cv2.resize(rgbImage, (64,64))
    flippedImage = cv2.flip(rgbImage, flipCode=1) # flip horizontally
    no_vehicle_images_list.append(rgbImage)
    no_vehicle_images_list.append(flippedImage)
    
for imagePath in no_vehicle_image_arr2:
    readImage = cv2.imread(imagePath)
    rgbImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
    rgbImage = cv2.resize(rgbImage, (64,64))
    no_vehicle_images_list.append(rgbImage)
    
    
print("Reading of non vehicle images done...")

print("No of Vehicles images: ", len(vehicle_images_list))
print("No of non vehicle images: ", len(no_vehicle_images_list))
#                       ENDING OF READING IMAGES FROM FILES

#                       FEATURE EXTRACTION FROM IMAGES

def GetFeaturesFromHog(image, orient, cellsPerBlock, pixelsPerCell,
                       visualise=False, feature_vector_flag=True):
    'To extract hog feature using hog() from YUV image'
    hog_features = hog(image, orientations=orient,
                                  pixels_per_cell=(pixelsPerCell, pixelsPerCell),
                                  cells_per_block=(cellsPerBlock, cellsPerBlock),
                                  visualize=visualise, feature_vector= feature_vector_flag, block_norm='L2-Hys')
    return hog_features

def ExtractFeatures(images, orientation=9, cellsPerBlock=2, pixelsPerCell=16, convertColorSpace=True):
    'Arrangin features into horizontal stack extracted from hog()'
    featureList = []
    for image in images:
        if(convertColorSpace):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        feature1 = GetFeaturesFromHog(image[:,:,0], orientation, cellsPerBlock, pixelsPerCell)
        feature2 = GetFeaturesFromHog(image[:,:,1], orientation, cellsPerBlock, pixelsPerCell)
        feature3 = GetFeaturesFromHog(image[:,:,2], orientation, cellsPerBlock, pixelsPerCell)

        # Arranging feature into horizontal stack for machine learning model training set
        featureListing = np.hstack((feature1, feature2, feature3))
        featureList.append(featureListing)
    return featureList

vehicleFeatures = ExtractFeatures(vehicle_images_list)
nonVehicleFeatures = ExtractFeatures(no_vehicle_images_list)

''' Arranging features into vertical stack for training the machine learning model '''
featureList = np.vstack([vehicleFeatures, nonVehicleFeatures])
print("Shape of features list is ", featureList.shape)
labelList = np.concatenate([np.ones(len(vehicleFeatures)), np.zeros(len(nonVehicleFeatures))])
print("Shape of labels list is ", labelList.shape)

#                       DATA PREPROCESSING

# train and test split of data
X_train, X_test, Y_train, Y_test = train_test_split(featureList, labelList, test_size=0.2, shuffle=True)


#                       TRAINING THE LINEAR SUPPORT VECTOR CLASSIFIER MODEL OF MACHINE LEARNING

svcClassifier  = LinearSVC()
svcClassifier.fit(X_train, Y_train) # training the classifier
print("Support Vector Classifier Trained...")
print("Accuracy of trained svm classifier is ", svcClassifier.score(X_test, Y_test))


#                    START DETECTING MOVING OBJECTS IN IMAGES

video = cv2.VideoCapture('camera.mp4') # Capturing video from camera

# Initialize the first frame in the video stream
firstFrame = None
count = 0

# loop over the frames of the video
while True:
    ret, frame = video.read()

    if frame is None:
        break


    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    frame = cv2.GaussianBlur(frame, (5,5),0)
    movingObject = frame.copy() # Generating copy of image for future use

    # Edge detection using canny edge detection method
    edges = cv2.Canny(frame, 100,200)

        
    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = edges
        continue

    if count > 20:
        firstFrame = edges
        count = 0
        continue
    else:
        count += 1

    ''' bitwise_and with firstFrame will give only static
    elements edges in white color which will be further changed into black
    for another bitwise_and to get moving elements'''
    maskImage = cv2.bitwise_and(firstFrame, edges)
    
    ''' Inversion after thresholding
    to make lines black and background white so when
    we do bitwise_and with real edges image, it will only give
    moving elements by setting other all to zero or black'''
    maskImage = cv2.threshold(maskImage, 254, 255, cv2.THRESH_BINARY_INV)[1]

    ''' bitwise_and with edges and maskImage will give a moving elements image'''
    carImage = cv2.bitwise_and(edges, maskImage)

    ''' Closing morphological transformation will close black hole in white lines
    of edges of moving elements to help in making better contours'''
    carImage = cv2.morphologyEx(carImage, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    #carImage = cv2.dilate(carImage, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))

    # Finding contours of moving element image
    cnts = cv2.findContours(carImage.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Drawing contours in the image for visualization
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        (x,y,w,h) = cv2.boundingRect(c)

        # Extracting moving object image and detecting cars
        clippedImage = frame[y:y+h, x:x+w]
        clippedImage1 = cv2.resize(clippedImage, (64,64))
        clippedFeature = ExtractFeatures([clippedImage1])
        predictedOutput = svcClassifier.predict([clippedFeature[0]])

        if (predictedOutput == 1):
            cv2.rectangle(movingObject, (x,y), (x+w,y+h),(0,255,0),2)
            #cv2.imshow('1 - detected car', clippedImage)
        #else:
        #    cv2.imshow('0 - Not detected car', clippedImage)

        
    cv2.imshow('Moving Object Detected Frame', movingObject)
      
    
    k = cv2.waitKey(30)
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
