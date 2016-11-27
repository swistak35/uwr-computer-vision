import scipy as sc
import scipy.io
import numpy as np
import cv2
from matplotlib import pyplot as plt

def solveWithScale(mtx, kScale):
    [_, _, r] = np.linalg.svd(mtx)
    vecUnscaled = r[-1]
    vec = vecUnscaled * (kScale / np.linalg.norm(vecUnscaled[0:3]))
    return vec

def task1():
    Po1Tld = sc.io.loadmat("data/Sport_cam.mat")['pml']
    Po2Tld = sc.io.loadmat("data/Sport_cam.mat")['pmr']

    Po1 = Po1Tld[:,0:3]
    Po2 = Po2Tld[:,0:3]
    p1Tld = Po1Tld[:,3]
    p2Tld = Po2Tld[:,3]
    c1 = -np.linalg.inv(Po1).dot(p1Tld)
    c2 = -np.linalg.inv(Po2).dot(p2Tld)
    alphaU = np.linalg.norm(np.cross(Po1[0], Po1[2]))
    alphaV = np.linalg.norm(np.cross(Po1[1], Po1[2]))

    f1f2 = np.cross(Po1[2], Po2[2])
    a3 = solveWithScale(np.array([
            [ c1[0],   c1[1],   c1[2],   1.0 ],
            [ c2[0],   c2[1],   c2[2],   1.0 ],
            [ f1f2[0], f1f2[1], f1f2[2], 0.0 ] # f1f2
        ]), 1.0)

    a2 = solveWithScale(np.array([
            [ c1[0], c1[1], c1[2], 1.0   ],
            [ c2[0], c2[1], c2[2], 1.0   ],
            [ a3[0], a3[1], a3[2], a3[3] ]
        ]), alphaV)

    a1 = solveWithScale(np.array([
            [ c1[0], c1[1], c1[2], 1.0 ],
            [ a2[0], a2[1], a2[2], a2[3] ],
            [ a3[0], a3[1], a3[2], a3[3] ]
        ]), alphaU)

    b1 = solveWithScale(np.array([
            [ c2[0], c2[1], c2[2], 1.0 ],
            [ a2[0], a2[1], a2[2], a2[3] ],
            [ a3[0], a3[1], a3[2], a3[3] ]
        ]), alphaU)

    Pn1Tld = np.array([ a1, a2, a3 ])
    Pn2Tld = np.array([ b1, a2, a3 ])

    print(Pn1Tld)
    print(Pn2Tld)

def task2():
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
    task1()
    task2()

run()
