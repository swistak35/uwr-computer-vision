# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:54:29 2016

@author: francho
"""

#Lab1
import numpy as np
import numpy.linalg
import scipy as sc
import scipy.linalg
import imageio as imio
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math



#Load 3D Points
#Load 2d Points1
#Load 2d Points2

def readPoints(filename,flag):
    points = []
    f = open(filename,'r')
    for line in f:
        line = line.split() # to deal with blank
        if line:            # lines (ie skip them)
            if flag == 'int':
                line = [int(i) for i in line]
            if flag == 'float':
                line = [float(i) for i in line]
            points.append(line)
    return points

pts2AN = readPoints("data/task12/pts2d-norm-pic_a.txt","float")
pts3DN = readPoints("data/task12/pts3d-norm.txt","float")

pts2A = readPoints("data/task12/pts2d-pic_a.txt","int")
pts2B = readPoints("data/task12/pts2d-pic_b.txt","int")
pts3D = readPoints("data/task12/pts3d.txt","float")

resultOfTask1 = np.array([
    [-0.4583, 0.2947, 0.0139, -0.0040],
    [0.0509, 0.0546, 0.5410, 0.0524],
    [-0.1090, -0.1784, 0.0443, -0.5968]
])

##############################

def createSystemForP(pts2d,pts3d):
    """
    Take two lists of correspondent pooints and creates a system of equations
    in the form of a matrix A
    """
    # assert that there's equal amount of points in pts2d and pts3d
    pointsPairs = zip(pts2d, pts3d)
    a = np.array([[
        pt3d[0], pt3d[1], pt3d[2], 1,
        0, 0, 0, 0,
        -pt2d[0]*pt3d[0], -pt2d[0]*pt3d[1], -pt2d[0]*pt3d[2], -pt2d[0],
        0, 0, 0, 0,
        pt3d[0], pt3d[1], pt3d[2], 1,
        -pt2d[1]*pt3d[0], -pt2d[1]*pt3d[1], -pt2d[1]*pt3d[2], -pt2d[1] ] for (pt2d, pt3d) in pointsPairs ])
    # assert shape == (pointsPairs, 24)
    return a.reshape(len(pointsPairs) * 2, 12)

def buildProjectionMatrix(rotationMtx, translationVector):
    translationVecAsColumn = np.array([[i] for i in translationVector])
    return np.concatenate((rotationMtx, translationVecAsColumn), axis=1)

def solveForP(A):
    """Given a system of linear equiations Ab = 0
       Solves the system using Linear Least Squares Minimization
       Using SVD numpy.linalg.svd
       For 3x4 matrix P
    """
    [l, s, r] = np.linalg.svd(A)
    P = r[-1].reshape(3,4)
    # Why r[-1] (last row)? Shouldn't it be r[:,-1] (last column)?
    # Or try using lstsq maybe?
    return P

def calculateP(pts2A, pts3D):
    A = createSystemForP(pts2A,pts3D)
    P = solveForP(A)
    return P

def KRTfromP(P):
    "given P decomposes it into K,R, and T"
    K, R = sc.linalg.rq(P[:,0:-1])
    T = np.linalg.inv(K).dot(P[:, -1])
    return (K,R,T)

def error(P,p2,p3D):
    error = 0.
    for i in range(20):
        p3H = p3D[i][0:3]
        p3H.append(1.0)
        p2h = np.dot(P,p3H)
        p2h = p2h/p2h[2]
        diff = (p2[i][0] - p2h[0])*(p2[i][0] - p2h[0]) + ((p2[i][1] - p2h[1])*(p2[i][1] - p2h[1]))
        error = error + math.sqrt(diff)
    print (error)

def calibrate(pts2A,pts3D):
    "returns P adn it's decomposition"
    P = calculateP(pts2A, pts3D)
    K,R,T = KRTfromP(P)
    return (P, K, R, T)

#////////////////////////////////////////////////

def task1():
    P,K,R,T = calibrate(pts2AN,pts3DN)
    error(P,pts2AN,pts3DN)
    P,K,R,T = calibrate(pts2A,pts3D)
    error(P,pts2A,pts3D)
    print (P)
    print (K)
    print (R)
    print (T)

def realTask1():
    P = calculateP(pts2AN, pts3DN)
    print (P)
    error(P, pts2AN, pts3DN)
    P = calculateP(pts2A, pts3D)
    print (P)
    error(P, pts2A, pts3D)

def task2():
    P,K,R,T = calibrate(pts2AN,pts3DN)
    print "Task 2"
    print "Projection matrix:"
    print (P)
    print (K.dot(buildProjectionMatrix(R, T)))
    print "Intrinsic matrix:"
    print (K)
    print "Rotation matrix:"
    print (R)
    print "Translation vector"
    print (T)


    print "Camera center: "
    C = np.linalg.inv(R).dot(-T)
    print (C)
    # Alternative way
    # C2 = -np.linalg.inv(R).dot(np.linalg.inv(K)).dot(P[:,-1])
    # print (C2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in pts3DN], [p[1] for p in pts3DN], [p[2] for p in pts3DN], marker='o')
    ax.scatter(C[0], C[1], C[2], marker = '^')
    # ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3], label='hehe', projection = '3d', alpha = 0.5, adjustable = 'box-forced')
    # ax.plot([10, 10], [20, 20], [30, 30])
    # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # z = np.linspace(-2, 2, 100)
    # r = z**2 + 1
    # x = r * np.sin(theta)
    # y = r * np.cos(theta)
    # ax.plot(x, y, z, label='parametric curve')

    # ax2 = fig.add_subplot(111, projection='3d', label = 'foo')
    # ax2.grid(b = False)
    # ax2.set_frame_on(False)
    # ax2.set_axis_off()

    plt.show()

realTask1()
task2()
