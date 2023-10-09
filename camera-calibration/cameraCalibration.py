import numpy as np
import cv2 as cv
import glob

#find chessboard corners - object points and image points#

chessBoardSize = (24, 17)
frameSize = (1440, 1080)

#termination criteria 
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ... (6,5,0)
objp = np.zeros((chessBoardSize[0] * chessBoardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

#arrays to store objects points and image points from all the images
objPoints = [] #3d points in real world space
imgPoints = [] #2d points in image plane

images = glob.glob('*.png')

for image in images:
  img = cv.imread(image)
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  #find the chess board corners
  ret, corners = cv.findChessboardCorners(gray, chessBoardSize, None)

  #if found, add object points, image points (after refining them)
  if ret == True:
    objPoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgPoints.append(corners)

  #draw and display the corners 
  cv.drawChessboardCorners(img, chessBoardSize, corners2, ret)
  cv.imshow('img', img)
  cv.waitKey(1000)

cv.destroyAllWindows()

#calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

#undistortion
img = cv.imread('cali5.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

#undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

#crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

#undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

#crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

#reprojection error
mean_error = 0

for i in range(len(objPoints)):
    imgpoints2, _ = cv.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgPoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objPoints)) )