from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import cv2
import numpy as np

filename = "f_r_1.png"

image = cv2.imread(filename)

shape = image.shape
image = cv2.resize(image,(shape[0]//3,shape[1]//6)) # resize

shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,2)

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((2, 2), np.uint8)
closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)

print(thresh[1])
 
# show the output image
cv2.imshow("Output", image)
cv2.imshow("Morph (closing)", closing)
cv2.imshow("Morph", opening)
cv2.waitKey(0)