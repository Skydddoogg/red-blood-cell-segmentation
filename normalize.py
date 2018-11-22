import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

def read_file():
    filename = "f_r_1.png"
    image = cv2.imread(filename)
    image_shape = image.shape
    image_resize_bgr = cv2.resize(image, (image_shape[0]//3, image_shape[1]//6))
    image_resize_gray = cv2.cvtColor(image_resize_bgr, cv2.COLOR_BGR2GRAY)
    return image_resize_bgr, image_resize_gray

def segment(thresh):
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    return labels

def plot_label_to_img(image, gray, labels):

    arranaged_labels = []

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
    
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
    
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > 10:
            cv2.circle(image, (int(x), int(y)), int(r), (255, 0, 0), 2)

        arranaged_labels.append((x, y, r))
        # print(((x, y), r))
        
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

    return arranaged_labels

def merge_intervals(t1, t2):
    return ((t1[0] + t2[0])//2, (t1[1] + t2[1])//2, (t1[2] + t2[2])//2)


def main():
    image_resize_bgr, image_resize_gray = read_file() # Read file, resize and convert to grayscale
    clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_resize_gray = clache.apply(image_resize_gray)
    _, image_resize_bin_1 = cv2.threshold(image_resize_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # image_resize_bin_1 = cv2.morphologyEx(image_resize_bin_1, cv2.MORPH_OPEN, np.ones((3, 3), dtype=int))
    # image_resize_bin_1 = cv2.morphologyEx(image_resize_bin_1, cv2.MORPH_DILATE, np.ones((3, 3), dtype=int), iterations= 3)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(image_resize_bin_1, cv2.MORPH_OPEN, kernel, iterations = 2)
    eroded = cv2.erode(opening, None, iterations = 4)
    # cv2.imshow("opening", opening)
    # cv2.imshow("eroded", eroded)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    labels = segment(eroded)
    new_labels = plot_label_to_img(image_resize_bgr, image_resize_gray, labels)

    # merged = []
    # check = []

    # for i in range(0, len(new_labels)):

    #     for j in range(0, len(new_labels)):

    #         if i in check:
    #             continue

    #         # Find distance
    #         a = np.array((new_labels[i][0], new_labels[i][1]))
    #         b = np.array((new_labels[j][0], new_labels[j][1]))
    #         dist = np.linalg.norm(a-b)

    #         if dist > 50:
    #             merged.append((new_labels[i][0], new_labels[i][1], new_labels[i][2]))
    #             continue
    #         else:
    #             c1 = (new_labels[i][0], new_labels[i][1], new_labels[i][2])
    #             c2 = (new_labels[j][0], new_labels[j][1], new_labels[j][2])
    #             merged.append(merge_intervals(c1, c2))
    #             check.append(j)
        
    # print(len(merged))
main()
