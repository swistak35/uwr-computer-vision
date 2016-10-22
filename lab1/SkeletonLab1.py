# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:54:29 2016

@author: francho
"""

#Lab1
import numpy as np
import scipy as sc
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
    
pts2AN = readPoints("data/task2/pts2d-norm-pic_a.txt","float")
pts3DN = readPoints("data/task2/pts3d-norm.txt","float")        
        
pts2A = readPoints("data/task2/pts2d-pic_a.txt","int")
pts2B = readPoints("data/task2/pts2d-pic_b.txt","int")
pts3D = readPoints("data/task2/pts3d.txt","float")

##############################

def createSystemForP(pts2d,pts3d):
    """
    Take two lists of correspondent pooints and creates a system of equations
    in the form of a matrix A
    """
    return A

def solveForP(A):
    """Given a system of linear equiations Ab = 0
       Solves the system using Linear Least Squares Minimization
       Using SVD numpy.linalg.svd
       For 3x4 matrix P
    """
    return P

def KRTfromP(P):
    "given P decomposes it into K,R, and T"
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
    A = createSystemForP(pts2A,pts3D)
    P = solveForP(A)    
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


task1()