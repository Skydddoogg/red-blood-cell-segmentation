import cv2
import numpy as np

filename = 'ring_form.png'
img = cv2.imread(filename)
shape = img.shape
img = cv2.resize(img,(shape[0]//5,shape[1]//5)) # resize

# average filtering
kernel = np.ones((3,3),np.float32)/9 # init kernel
dst = cv2.filter2D(img,-1,kernel) # apply filter

# increase contrast
lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


B,G,R = cv2.split(final)

# Y = 16 + (65.481*R + 128.553*G + 24.966*B)
# Cb = 128 + (-37.797*R - 74.203*G + 112.00*B)
# Cr = 128 + (112.00*R - 93.786*G - 18.214*B)

# conversed_img = cv2.merge((Y, Cb, Cr))

imgYCC = cv2.cvtColor(final, cv2.COLOR_BGR2YCR_CB)

Y, Cb, Cr = cv2.split(imgYCC)


# a = np.array([[24.966, 128.553, 65.481],[112.00, -74.203, -37.797],[-18.214, -93.786, 112.00]])
# print(a*final)

# show image
# cv2.imshow('filtered + increase contrast', final)
# cv2.imshow('original', img)
# cv2.imshow('without lib', conversed_img)
# cv2.imshow('with lib', imgYCC)

cv2.imshow('Y', Y)
cv2.imshow('Cb', Cb)
cv2.imshow('Cr', Cr)
cv2.waitKey(0)
cv2.destroyAllWindows()
