from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import cv2
import numpy as np

filename = "ring_form.png"

image = cv2.imread(filename)

shape = image.shape
image = cv2.resize(image,(shape[0]//3,shape[1]//6)) # resize

shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

print(thresh[1])
 
# show the output image
cv2.imshow("Output", image)
cv2.imshow("Thresh", thresh[1])
cv2.imshow("Adaptive", th3)
cv2.waitKey(0)
