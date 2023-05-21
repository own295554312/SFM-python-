import cv2
import numpy as np

img1 = cv2.imread("D:\Program Files\MATLAB\R2022b\toolbox\vision\visiondata\structureFromMotion\image1.jpg")
img2 = cv2.imread("D:\Program Files\MATLAB\R2022b\toolbox\vision\visiondata\structureFromMotion\image2.jpg")
data = np.load("D:\Program Files\MATLAB\R2022b\toolbox\vision\visiondata\structureFromMotion\cameraParams.mat", allow_pickle=True)

cameraParams = data['cameraParams']

# 获得内参矩阵
intrinsics = cameraParams.Intrinsics
K = intrinsics.K
# 对第一张图像进行去畸变校正
img1 = cv2.undistort(img1, K, intrinsics.D)
img2 = cv2.undistort(img2, K, intrinsics.D)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

border = 50
roi = (border, border, img1.shape[1] - 2 * border, img1.shape[0] - 2 * border)
surf = cv2.xfeatures2d.SURF_create(nOctaves=8)
keypoints1, descriptors1 = surf.detectAndCompute(img1, None, roi)
keypoints2, descriptors2 = surf.detectAndCompute(img2, None, roi)

img1_keypoints = cv2.drawKeypoints(img1, keypoints1[:100], None)
img2_keypoints = cv2.drawKeypoints(img2, keypoints2[:100], None)

cv2.imshow("Image 1 Keypoints", img1_keypoints)
cv2.imshow("Image 2 Keypoints", img2_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
matches = matcher.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:100]

matchedPoints1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
matchedPoints2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

F, mask = cv2.findFundamentalMat(matchedPoints1, matchedPoints2, cv2.FM_RANSAC, 0.7, 0.99)

E = np.dot(np.dot(K.T, F), K)

U, _, V = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

R1 = np.dot(np.dot(U, W), V)
R2 = np.dot(np.dot(U, W.T), V)

if np.linalg.det(R1) < 0:
    R1 = -R1
if np.linalg.det(R2) < 0:
    R2 = -R2

Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
A = np.dot(np.dot(U, Z), U.T)

