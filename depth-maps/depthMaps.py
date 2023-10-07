import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

left_image = cv.imread('depth-maps\/tsukuba_l.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('depth-maps\/tsukuba_r.png', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=21)
#for each pixel algoritmh will find the best disparity
#larger block size implies smoother, though less accurate disparity
depth = stereo.compute(left_image, right_image)

cv.imshow("Left", left_image)
cv.imshow("Right", right_image)

plt.imshow(depth)
plt.axis('off')
plt.show()