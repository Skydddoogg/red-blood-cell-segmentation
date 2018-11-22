from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from scipy.ndimage import label
import argparse
import cv2
import numpy as np

filename = "f_r_3.png"

image = cv2.imread(filename)
image_shape = image.shape
image_resize = cv2.resize(image, (image_shape[0]//3, image_shape[1]//6))

# shifted = cv2.pyrMeanShiftFiltering(image_resize, 21, 51)

gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

# gray2 = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# laplacian = cv2.Laplacian(gray2,cv2.CV_64F)

th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,2)

kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

# for i in range(20):
#     opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# dilation = cv2.dilate(opening,kernel,iterations = 1)
# erosion = cv2.erode(dilation,kernel,iterations = 4)
# opening2 = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
# dilation2 = cv2.dilate(opening2,kernel,iterations = 3)

# cv2.imshow("Gray", gray)
# cv2.imshow("Threshold", th3)
# cv2.imshow("Threshold2", thresh[1])
cv2.imshow("Closing", closing)
# cv2.imshow("Threshold (Normal)", th1)
# cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)
