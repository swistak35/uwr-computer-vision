import scipy as sc
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import os.path
from matplotlib import pyplot as plt

def KRTfromP(P):
    "given P decomposes it into K,R, and T"
    K, R = sc.linalg.rq(P[:,0:-1])
    T = np.linalg.inv(K).dot(P[:, -1])
    return (K,R,T)

def from2Homogenous(coords):
    assert(coords.shape[1] == 3)
    return coords[:,0:2] / coords[:,2][:,None]

def toHomogenous(coords):
    return np.hstack((coords, np.ones(coords.shape[0])[:,None]))

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

def rectifyImage(T1):
    filename = "data/Sport0.png"
    image = scipy.ndimage.imread(filename)
    cx,cy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    r = np.stack((cx,cy), axis=2).transpose((1, 0, 2)).reshape((-1,2))
    r = np.fliplr(r)
    imager = image[:,:,0]
    imageg = image[:,:,1]
    imageb = image[:,:,2]
    imageShape = image.shape[0:2]

    homor = toHomogenous(r)
    transformedHomoR = T1.dot(homor.T).T
    heteroPoints = from2Homogenous(transformedHomoR)

    xMin, yMin = np.amin(heteroPoints, axis = 0)
    xMax, yMax = np.amax(heteroPoints, axis = 0)

    flippedPoints = np.fliplr(heteroPoints)
    points4 = flippedPoints

    # scipy.ndimage.interpolation.h)


    # intrInv = np.linalg.inv(intrinsicMtx)
    # normalizedHomopoints = intrInv.dot(endhomor.T)
    # projectedPoints = from2Homogenous(normalizedHomopoints.T)
    # correctedPoints = undistortedPoints(projectedPoints, distortion)
    # homopoints2 = toHomogenous(correctedPoints)
    # points3 = intrinsicMtx.dot(homopoints2.T)
    # points4 = from2Homogenous(points3.T)
    # points4 = np.fliplr(points4)
    mappedPointsR = sc.ndimage.map_coordinates(imager, points4.T, order=3).reshape(imageShape)
    mappedPointsG = sc.ndimage.map_coordinates(imageg, points4.T, order=3).reshape(imageShape)
    mappedPointsB = sc.ndimage.map_coordinates(imageb, points4.T, order=3).reshape(imageShape)
    newimage = np.stack((mappedPointsR, mappedPointsG, mappedPointsB), axis=-1)

    sc.misc.imsave(mkPath(filename, "-correctedimage"), newimage)


def rectify(Po1Tld, Po2Tld):
    Po1 = Po1Tld[:,0:3]
    Po2 = Po2Tld[:,0:3]
    p1Tld = Po1Tld[:,3]
    p2Tld = Po2Tld[:,3]
    c1 = -np.linalg.inv(Po1).dot(p1Tld)
    c2 = -np.linalg.inv(Po2).dot(p2Tld)

    Ko1, Ro1, To1 = KRTfromP(Po1Tld)
    Ko2, Ro2, To2 = KRTfromP(Po2Tld)

    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(Ro1[2], r1)
    r3 = np.cross(r1, r2)
    Rn = np.array([ r1, r2, r3 ])

    Tn1 = -Rn.dot(c1)
    Tn2 = -Rn.dot(c2)
    Pn1 = Ko1.dot(np.hstack((Rn, Tn1.reshape(3,1))))
    Pn2 = Ko1.dot(np.hstack((Rn, Tn2.reshape(3,1))))

    T1 = Pn1[:,0:3].dot(np.linalg.inv(Po1))
    T2 = Pn2[:,0:3].dot(np.linalg.inv(Po2))

    return (Pn1, Pn2, T1, T2)

def task12():
    print("=== Task 1")
    Po1Tld = sc.io.loadmat("data/Sport_cam.mat")['pml']
    Po2Tld = sc.io.loadmat("data/Sport_cam.mat")['pmr']

    Pn1, Pn2, T1, T2 = rectify(Po1Tld, Po2Tld)
    print("== Pn1")
    print(Pn1)
    print("== Pn2")
    print(Pn2)
    print("== T1")
    print(T1)
    print("== T2")
    print(T2)

    rectifyImage(T1)

def task3():
    imgL = cv2.imread("data/rectified/Sport_R_0.png", 0)
    imgR = cv2.imread("data/rectified/Sport_R_1.png", 0)

    # stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    disp = stereo.compute(imgL,imgR).astype(np.float32) / 16.0

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    # cv2.imshow('disparity', disparity / 16.0 )
    cv2.waitKey()

    # plt.imshow(disparity,'gray')
    # plt.show()

def run():
    task12()
    # task3()

run()
