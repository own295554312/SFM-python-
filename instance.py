import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from vision.internal.path import toolboxdir
from vision.internal.visiondata import upToScaleReconstructionImages

imageDir = os.path.join(toolboxdir('vision'), 'visiondata', 'upToScaleReconstructionImages')
images = cv2.data.build_image_sequence(imageDir)
I1 = images[0]
I2 = images[1]
plt.figure()
plt.imshow(np.hstack((I1, I2)))
plt.title('Original Images')

data = np.load('sfmCameraIntrinsics.npz')
intrinsics = data['intrinsics']

I1 = cv2.undistort(I1, intrinsics['cameraMatrix'], intrinsics['distCoeffs'])
I2 = cv2.undistort(I2, intrinsics['cameraMatrix'], intrinsics['distCoeffs'])
plt.figure()
plt.imshow(np.hstack((I1, I2)))
plt.title('Undistorted Images')

imagePoints1 = cv2.goodFeaturesToTrack(cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY), maxCorners=150, qualityLevel=0.1, minDistance=5)
plt.figure()
plt.imshow(I1)
plt.title('150 Strongest Corners from the First Image')
plt.plot(imagePoints1[:, 0, 0], imagePoints1[:, 0, 1], 'ro')

tracker = cv2.TrackerKCF_create()
tracker.init(I1, imagePoints1)

ok, imagePoints2 = tracker.update(I2)
matchedPoints1 = imagePoints1[ok]
matchedPoints2 = imagePoints2[ok]

plt.figure()
plt.imshow(cv2.drawMatches(I1, imagePoints1, I2, imagePoints2, None, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
plt.title('Tracked Features')

E, mask = cv2.findEssentialMat(matchedPoints1, matchedPoints2, intrinsics['cameraMatrix'], method=cv2.RANSAC, prob=0.999, threshold=1.0)
inlierPoints1 = matchedPoints1[mask.ravel().astype(bool)]
inlierPoints2 = matchedPoints2[mask.ravel().astype(bool)]

plt.figure()
plt.imshow(cv2.drawMatches(I1, inlierPoints1, I2, inlierPoints2, None, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
plt.title('Epipolar Inliers')

retval, R, T, mask = cv2.recoverPose(E, inlierPoints1, inlierPoints2, intrinsics['cameraMatrix'])
relPose = np.hstack((R, T))

border = 30
roi = (border, border, I1.shape[1] - 2 * border, I1.shape[0] - 2 * border)
imagePoints1 = cv2.goodFeaturesToTrack(cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY), maxCorners=150, qualityLevel=0.001, minDistance=5, mask=np.zeros_like(I1, dtype=np.uint8))
tracker = cv2.TrackerKCF_create()
tracker.init(I1, imagePoints1)

ok, imagePoints2 = tracker.update(I2)
matchedPoints1 = imagePoints1[ok]
matchedPoints2 = imagePoints2[ok]

camMatrix1 = np.eye(3, 4)
camMatrix2 = np.hstack((intrinsics['cameraMatrix'], np.dot(intrinsics['cameraMatrix'], relPose[:, :3])))

points4D = cv2.triangulatePoints(camMatrix1, camMatrix2, matchedPoints1.T, matchedPoints2.T)
points3D = points4D[:3] / points4D[3]

numPixels = I1.shape[0] * I1.shape[1]
allColors = np.reshape(I1, (numPixels, 3))
colorIdx = np.ravel_multi_index((np.round(matchedPoints1[:, 1]), np.round(matchedPoints1[:, 0])), (I1.shape[0], I1.shape[1]))
color = allColors[colorIdx]

ptCloud = np.zeros((points3D.shape[1], 6))
ptCloud[:, :3] = points3D.T
ptCloud[:, 3:] = color / 255.0

cameraSize = 0.3
plt.figure()
plt.plot([0], [0], 'ro', markersize=cameraSize * 100, label='1')
plt.plot(relPose[0, 3], relPose[1, 3], 'bo', markersize=cameraSize * 100, label='2')
plt.grid()
plt.axis('equal')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Camera Locations and Orientations')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ptCloud[:, 0], ptCloud[:, 1], ptCloud[:, 2], c=ptCloud[:, 3:], s=45)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Point Cloud')

globe = cv2.Sphere3D()

plt.figure()
globe.plot()
plt.title('Estimated Location and Size of the Globe')

scaleFactor = 10 / globe.Radius
ptCloud[:, :3] *= scaleFactor
relPose[:, 3] *= scaleFactor

cameraSize = 2
plt.figure()
plt.plot([0], [0], 'ro', markersize=cameraSize * 100, label='1')
plt.plot(relPose[0, 3], relPose[1, 3], 'bo', markersize=cameraSize * 100, label='2')
plt.grid()
plt.axis('equal')
plt.xlabel('x-axis (cm)')
plt.ylabel('y-axis (cm)')
plt.title('Metric Reconstruction of the Scene')
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ptCloud[:, 0], ptCloud[:, 1], ptCloud[:, 2], c=ptCloud[:, 3:], s=45)
ax.set_xlabel('X-axis (cm)')
ax.set_ylabel('Y-axis (cm)')
ax.set_zlabel('Z-axis (cm)')
ax.set_title('Point Cloud in Centimeters')

plt.show()

