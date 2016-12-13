import scipy as sc
import scipy.io
import scipy.ndimage
import numpy as np
import cv2
import os.path
from matplotlib import pyplot as plt
from plyfile import PlyData, PlyElement

np.set_printoptions(threshold=np.nan)

DEBUG = False

def debug(arg):
    if DEBUG:
        print(arg)

def renderPlyFile(XX, filename):
    if len(XX) > 0:
        points = np.array(zip(XX[:,0].ravel(), XX[:,1].ravel(), XX[:,2].ravel()),dtype=[('x','f4'), ('y','f4'),('z', 'f4')])
        el = PlyElement.describe(points, 'vertex')
        PlyData([el]).write(filename)
    else:
        print("Nothing rendered into file '%s' because points set was empty" % filename)

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
    return (homor[1] - 8.0, homor[0] - 154.0)

def computeNeededShape(imageShape, mtx):
    (height, width) = imageShape
    debug(imageShape)
    cornerTopLeft = np.asarray([0.0, 0.0, 1.0])
    cornerTopRight = np.asarray([width, 0.0, 1.0])
    cornerBottomLeft = np.asarray([0.0, height, 1.0])
    cornerBottomRight = np.asarray([width, height, 1.0])
    pts = np.array([ cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight ])
    pts = mtx.dot(pts.T).T
    pts = pts[:,0:2] / pts[:,2:3]
    # xMin, yMin = np.amin(pts, axis=0)
    # xMax, yMax = np.amax(pts, axis=0)
    xSpan, ySpan = np.ptp(pts, axis=0)

    return (ySpan.astype(np.int32), xSpan.astype(np.int32))

def rectifyImage(filename, T1):
    image = scipy.ndimage.imread(filename)
    cx,cy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    r = np.stack((cx,cy), axis=2).transpose((1, 0, 2)).reshape((-1,2))
    r = np.fliplr(r)
    imager = image[:,:,0]
    imageg = image[:,:,1]
    imageb = image[:,:,2]
    imageShape = image.shape[0:2]

    newImageShape = computeNeededShape(imageShape, T1)
    debug("newImageShape")
    debug(newImageShape)

    o = np.arange(newImageShape[0] * newImageShape[1]).reshape(newImageShape)

    T1inv = np.linalg.inv(T1)
    debug(imager.shape)
    mappedPointsR = sc.ndimage.interpolation.geometric_transform(imager, matrix_applier, output = o, output_shape = newImageShape, extra_arguments=(T1inv,))
    debug(mappedPointsR)

    coords = (0, 0)
    homor = T1inv.dot(np.array([coords[1], coords[0], 1.0]))
    homor = homor / homor[2]
    debug("== (0,0) transformed to:")
    debug((homor[1], homor[0]))

    coords = (0, 0)
    homor = T1.dot(np.array([coords[1], coords[0], 1.0]))
    homor = homor / homor[2]
    debug("== (0,0) transformed to:")
    debug((homor[1], homor[0]))

    sc.misc.imsave(mkPath(filename, "-correctedimage"), o)

def rectify(Po1Tld, Po2Tld):
    Po1 = Po1Tld[:,0:3]
    Po2 = Po2Tld[:,0:3]
    p1Tld = Po1Tld[:,3]
    p2Tld = Po2Tld[:,3]
    c1 = -np.linalg.inv(Po1).dot(p1Tld)
    c2 = -np.linalg.inv(Po2).dot(p2Tld)

    Ko1, Ro1, To1 = KRTfromP(Po1Tld)
    Ko2, Ro2, To2 = KRTfromP(Po2Tld)

    r1 = (c2 - c1) / np.linalg.norm(c2 - c1)
    r2 = np.cross(Ro1[2], r1)
    r3 = np.cross(r1, r2)
    Rn = np.array([ r1, r2, r3 ])

    Kn = (Ko1 + Ko2) / 2 # Isn't that enough? Why do we still need vertical displacement?
    # Kn = Ko1
    Kn[0,1] = 0.0

    Tn1 = -Rn.dot(c1)
    Tn2 = -Rn.dot(c2)
    Pn1 = Kn.dot(np.hstack((Rn, Tn1.reshape(3,1))))
    Pn2 = Kn.dot(np.hstack((Rn, Tn2.reshape(3,1))))

    T1 = Pn1[:,0:3].dot(np.linalg.inv(Po1))
    T2 = Pn2[:,0:3].dot(np.linalg.inv(Po2))

    baselineLength = np.linalg.norm(Tn1 - Tn2)

    return (Pn1, Pn2, T1, T2, Kn, baselineLength)

def computeDisparity():
    imgL = cv2.imread("data/rectified/Sport_R_0.png", 1)
    imgR = cv2.imread("data/rectified/Sport_R_1.png", 1)

    stereo = cv2.StereoSGBM_create(
        minDisparity = 32,
        numDisparities = 144,
        blockSize = 13,
        P1 = 2048,
        P2 = 8192,
        speckleWindowSize = 125,
        speckleRange = 3,
    )
    disp = stereo.compute(imgL, imgR)
    return disp

def run():
    print("=== Task 1")
    Po1Tld = sc.io.loadmat("data/Sport_cam.mat")['pml']
    Po2Tld = sc.io.loadmat("data/Sport_cam.mat")['pmr']

    Pn1, Pn2, T1, T2, Kn, baselineLength = rectify(Po1Tld, Po2Tld)
    print("== Pn1")
    print(Pn1)
    print("== Pn2")
    print(Pn2)
    print("== T1")
    print(T1)
    print("== T2")
    print(T2)

    print("=== Task 2")
    rectifyImage("data/Sport0.png", T1)
    rectifyImage("data/Sport1.png", T2)

    print("=== Task 3")
    disp = computeDisparity()

    # print(np.amin(disp))
    # print(np.amax(disp))
    cv2.imshow('disparity', disp.astype(np.float32) / np.amax(disp))
    cv2.waitKey()

    # print("=== Task 4")
    # focalX = Kn[0,0]
    # depth = np.full(disp.shape, baselineLength * focalX) / disp

    # cx,cy = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]))
    # r = np.stack((cx,cy), axis=2).transpose((1, 0, 2)).reshape((-1,2))
    # r = np.fliplr(r)
    # r2 = np.array([ (x, y, depth[y,x]) for (x,y) in r ])

    # KnInv = np.linalg.inv(Kn)
    # normalizedPoints = KnInv.dot(r2.T).T
    # renderPlyFile(normalizedPoints, "pc.ply")


run()
