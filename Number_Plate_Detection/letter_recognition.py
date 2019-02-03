import cv2
import numpy as np
import re
import pandas as pd

dataFile = None

# Reading data from file in string format
with  open('letter-recognition.data', mode='r') as file:
    dataFile = file.read()

# Changing data to list of strings
stringList = list(map(str, dataFile.split('\n')))
stringList = stringList[:-1]

# Changing data letter to ascii value and others- convert to int
for ind in range(len(stringList)):
    stringList[ind] = list(map(str, stringList[ind].split(',')))
    stringList[ind][0] = ord(stringList[ind][0])-ord('A')
    for i in range(1,len(stringList[ind])):
        stringList[ind][i]=int(stringList[ind][i])

# Transforming data into array
letterData = np.array(stringList)
letters, labels = letterData[:,1:], letterData[:,0]

a = np.array(letters[0]).astype(np.uint8)
a = a.reshape(4,4)
print(a)
cv2.imshow('T 19', a)
cv2.waitKey(0)

