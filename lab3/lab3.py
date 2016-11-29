import scipy as sc
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import os.path
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)

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

def matrix_applier(coords, mtx):
    homor = mtx.dot(np.array([coords[1], coords[0], 1.0]))
    homor = homor / homor[2]
    return (homor[1], homor[0])

def rectifyImage(T1):

    # T1 = np.array([
    #         [  9.34320999e-01,  -1.76269837e-02,   9.08629969e+00],
    #         [ -2.61206725e-02,   9.99853303e-01,  -1.65948291e+01],
    #         [ -1.48091221e-04,   6.28016064e-11,   1.06958634e+00]
    #     ])

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
    print(xMin, yMin)
    print(xMax, yMax)

    flippedPoints = np.fliplr(heteroPoints)
    points4 = flippedPoints

    # scipy.ndimage.interpolation.h)

    cx,cy = np.meshgrid(np.arange(800), np.arange(650))
    r = np.stack((cx,cy), axis=2).transpose((1, 0, 2)).reshape((-1,2))
    r = np.fliplr(r)
    # homor = toHomogenous(r)

    shape = (650, 950)
    o = np.arange(shape[0] * shape[1]).reshape(shape)

    T1inv = np.linalg.inv(T1)
    print(imager.shape)
    mappedPointsR = sc.ndimage.interpolation.geometric_transform(imager, matrix_applier, output = o, output_shape = shape, extra_arguments=(T1inv,))
    print(mappedPointsR)

    coords = (0, 0)
    homor = T1inv.dot(np.array([coords[1], coords[0], 1.0]))
    homor = homor / homor[2]
    print ("== (0,0) transformed to:")
    print (homor[1], homor[0])

    # intrInv = np.linalg.inv(intrinsicMtx)
    # normalizedHomopoints = intrInv.dot(endhomor.T)
    # projectedPoints = from2Homogenous(normalizedHomopoints.T)
    # correctedPoints = undistortedPoints(projectedPoints, distortion)
    # homopoints2 = toHomogenous(correctedPoints)
    # points3 = intrinsicMtx.dot(homopoints2.T)
    # points4 = from2Homogenous(points3.T)
    # points4 = np.fliplr(points4)
    # mappedPointsR = sc.ndimage.map_coordinates(imager, points4.T, order=3).reshape(imageShape)
    # mappedPointsG = sc.ndimage.map_coordinates(imageg, points4.T, order=3).reshape(imageShape)
    # mappedPointsB = sc.ndimage.map_coordinates(imageb, points4.T, order=3).reshape(imageShape)
    # newimage = np.stack((mappedPointsR, mappedPointsG, mappedPointsB), axis=-1)

    sc.misc.imsave(mkPath(filename, "-correctedimage"), o)

baseline = None
focalX = None
focalY = None

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

    print("Distance between two camera centres:")
    baseline = np.linalg.norm(Tn1 - Tn2)
    print(baseline)
    print(Ko1)
    print(Ko2)
    focalX = Ko1[0,0]
    focalY = Ko1[1,1]

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

    # window_size = 3
    # min_disp = 16
    # num_disp = 112-min_disp
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity = min_disp,
    #     numDisparities = num_disp,
    #     blockSize = 16,
    #     P1 = 8*3*window_size**2,
    #     P2 = 32*3*window_size**2,
    #     disp12MaxDiff = 1,
    #     uniquenessRatio = 10,
    #     speckleWindowSize = 100,
    #     speckleRange = 32
    # )
    # possible_numDisparities = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160]
    # possible_blockSize = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

    # 144 / 160 i 13 byly znosne
    # for c_numDisparity in possible_numDisparities:
    #     for c_blockSize in possible_blockSize:
    #         print("numDisparities=%d blockSize=%d" % (c_numDisparity, c_blockSize))
    #         stereo = cv2.StereoBM_create(numDisparities=c_numDisparity, blockSize=c_blockSize)
    #         disp = stereo.compute(imgL, imgR) / 16 #.astype(np.float32) #.astype(np.float32) #/ 16.0
    #         # print(disp)
    #         # cv2.imshow('disparity', disp)
    #         # cv2.waitKey()
    #         plt.imshow(disp,'gray')
    #         plt.show()
    # cv2.imshow('left', imgL)
    # cv2.imshow('disparity', disparity / 16.0 )

    # stereo = cv2.StereoBM_create(numDisparities=144, blockSize=13)
    # disp = stereo.compute(imgL, imgR) * 16 #/ 16.0 #.astype(np.float32) #.astype(np.float32) #/ 16.0
    # print(np.amin(imgL))
    # print(np.amax(imgL))
    # print(disp.dtype)
    # # print imgL
    # cv2.imshow('disparity', disp.astype(np.float32) / np.amax(disp))
    # cv2.waitKey()

    # imgL = cv2.imread("data/rectified/Sport_R_0.png", 1)
    # imgR = cv2.imread("data/rectified/Sport_R_1.png", 1)

    stereo = cv2.StereoSGBM_create(
        minDisparity = 32,
        numDisparities = 144,
        blockSize = 13,
        # P1 = 4*13**2,
        # P2 = 16*13**2,
        # disp12MaxDiff = 1,
        # uniquenessRatio = 10,
        speckleWindowSize = 75,
        speckleRange = 1,
        # fullDP = True
    )
    disp = stereo.compute(imgL, imgR)
    print(np.amin(disp))
    print(np.amax(disp))
    cv2.imshow('disparity', disp.astype(np.float32) / np.amax(disp))
    cv2.waitKey()


def run():
    task12()
    # task3()

run()
