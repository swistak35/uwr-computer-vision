import numpy as np
import PIL as pil
import Image, ImageDraw
import scipy.ndimage
import scipy.misc
from common import *

def loadCalibrationMatrix(f):
    rows = []
    for i in range(0,3): # 3 lines
        line = f.readline().split()
        rows.append([float(i) for i in line])
    rotationMtx = np.array(rows)
    translationVector = np.array([float(i) for i in f.readline().split()])
    return (rotationMtx, translationVector)

def loadCalibData(filename):
    f = open(filename,'r')
    [a,b,c,u0,v0] = f.readline().split()
    intrinsicMtx = np.array([
        [float(a), float(b), float(u0)],
        [0.0, float(c), float(v0)],
        [0.0, 0.0, 1.0]
        ])
    f.readline()
    distortion = [ float(k) for k in f.readline().split() ]
    f.readline()

    mtxs = []
    for i in range(0,5): # 5 matrices
        rotMtx, tVec = loadCalibrationMatrix(f)
        f.readline()
        projectionMatrix = buildProjectionMatrix(rotMtx, tVec)
        mtxs.append(projectionMatrix)

    return (distortion, intrinsicMtx, mtxs)

def loadModelPoints(filename):
    points = []
    f = open(filename, "r")
    for line in f:
        line = line.split()
        if line:
            points.append([float(line[0]), float(line[1])])
            points.append([float(line[2]), float(line[3])])
            points.append([float(line[4]), float(line[5])])
            points.append([float(line[6]), float(line[7])])
    return points

def drawPoints(filename, points, suffix = "-withpoints"):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for point in points:
        drawSmallCircle(draw, point[0], point[1])
    im.save(mkPath(filename, suffix), "GIF")

def savePictureWithPoints(filename, intrinsicMatrix, extrinsicMatrix, homopoints):
    projectionMatrix = intrinsicMatrix.dot(extrinsicMatrix)
    projectedHomopoints = projectionMatrix.dot(np.array(homopoints).T)
    projectedPoints = from2Homogenous(projectedHomopoints.T)
    drawPoints(filename, projectedPoints)

def correctedPoint(p, distortion):
    # assert shape == 2
    k1, k2 = distortion
    r2 = p[0]*p[0] + p[1]*p[1]
    coefficient = 1 + k1*r2 + k2*r2*r2
    return (p[0]*coefficient, p[1]*coefficient)

def savePictureWithCorrectedPoints(filename, intrinsicMtx, extrinsicMatrix, homopoints, distortion):
    points1 = [ extrinsicMatrix.dot(p) for p in homopoints ]
    projectedPoints = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in points1 ]
    correctedPoints = [ correctedPoint(p, distortion) for p in projectedPoints ]
    homopoints2 = [ (p[0], p[1], 1.0) for p in correctedPoints ]
    points3 = [ intrinsicMtx.dot(p) for p in homopoints2 ]
    points4 = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in points3 ]
    drawPoints(filename, points4, suffix = "-correctedpoints")

def correctImage(filename, distortion, intrinsicMtx, extrinsicMtx):
    image = scipy.ndimage.imread(filename)
    cx,cy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    r = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')
    imager = image[:,:,0]
    imageg = image[:,:,1]
    imageb = image[:,:,2]
    extrinsicMatrix = extrinsicMtx

    endhomor = np.array([ (p[0], p[1], 1.0) for p in r ])
    intrInv = np.linalg.inv(intrinsicMtx)
    normalizedHomopoints = intrInv.dot(endhomor.transpose())
    projectedPoints = np.array([ (hp[0] / hp[2], hp[1] / hp[2]) for hp in normalizedHomopoints.transpose() ])
    correctedPoints = np.array([ correctedPoint(p, distortion) for p in projectedPoints ])
    homopoints2 = np.array([ (p[0], p[1], 1.0) for p in correctedPoints ])
    points3 = np.array([ intrinsicMtx.dot(p) for p in homopoints2 ])
    points4 = np.array([ (hp[0] / hp[2], hp[1] / hp[2]) for hp in points3 ])
    mappedPointsR = scipy.ndimage.map_coordinates(imager, points4.transpose(), order=3).reshape(480, 640)
    mappedPointsG = scipy.ndimage.map_coordinates(imageg, points4.transpose(), order=3).reshape(480, 640)
    mappedPointsB = scipy.ndimage.map_coordinates(imageb, points4.transpose(), order=3).reshape(480, 640)
    newimage = np.stack((mappedPointsR, mappedPointsG, mappedPointsB), axis=-1)

    scipy.misc.imsave(mkPath(filename, "-correctedimage"), newimage)

def run():
    distortion, intrinsicMtx, extrinsicMtxs = loadCalibData("data/task34/Calib.txt")
    points = loadModelPoints("data/task34/Model.txt")
    homopoints = [ (p[0], p[1], 0.0, 1.0) for p in points ]

    # Undistorted pictures
    savePictureWithPoints("data/task34/UndistortIm1.gif", intrinsicMtx, extrinsicMtxs[0], homopoints)
    savePictureWithPoints("data/task34/UndistortIm2.gif", intrinsicMtx, extrinsicMtxs[1], homopoints)

    # Distorted pictures
    savePictureWithPoints("data/task34/CalibIm1.gif", intrinsicMtx, extrinsicMtxs[0], homopoints)
    savePictureWithPoints("data/task34/CalibIm2.gif", intrinsicMtx, extrinsicMtxs[1], homopoints)
    savePictureWithPoints("data/task34/CalibIm3.gif", intrinsicMtx, extrinsicMtxs[2], homopoints)
    savePictureWithPoints("data/task34/CalibIm4.gif", intrinsicMtx, extrinsicMtxs[3], homopoints)
    savePictureWithPoints("data/task34/CalibIm5.gif", intrinsicMtx, extrinsicMtxs[4], homopoints)

    # Distorted images with correct points
    savePictureWithCorrectedPoints("data/task34/CalibIm1.gif", intrinsicMtx, extrinsicMtxs[0], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm2.gif", intrinsicMtx, extrinsicMtxs[1], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm3.gif", intrinsicMtx, extrinsicMtxs[2], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm4.gif", intrinsicMtx, extrinsicMtxs[3], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm5.gif", intrinsicMtx, extrinsicMtxs[4], homopoints, distortion)

    # Undistort images
    correctImage("data/task34/CalibIm1.gif", distortion, intrinsicMtx, extrinsicMtxs[0])

    # Add undistorted points to undistorted images
    savePictureWithPoints("data/task34/CalibIm1-correctedimage.gif", intrinsicMtx, extrinsicMtxs[0], homopoints)

run()
