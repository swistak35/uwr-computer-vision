import numpy as np
import PIL as pil
import Image, ImageDraw
import os.path

def loadCalibrationMatrix(f):
    rows = []
    for i in range(0,3): # 3 lines
        line = f.readline().split()
        rows.append([float(i) for i in line])
    rotationMtx = np.array(rows)
    translationVector = np.array([float(i) for i in f.readline().split()])
    return (rotationMtx, translationVector)

def buildProjectionMatrix(rotationMtx, translationVector):
    translationVecAsColumn = np.array([[i] for i in translationVector])
    return np.concatenate((rotationMtx, translationVecAsColumn), axis=1)

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

def drawSmallCircle(draw, x, y):
    # Top left and bottom right corners
    colorPoint = 256
    colorCircle = 200
    width = 3
    draw.ellipse((x - width, y - width, x + width, y + width), fill = colorCircle)
    draw.point((x, y), fill = colorPoint)

def drawPoints(filename, points, suffix = "-withpoints"):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    # drawSmallCircle(draw, 100, 100)
    for point in points:
        drawSmallCircle(draw, point[0], point[1])
    basePath, extPath = os.path.splitext(filename)
    # im.show(draw)
    im.save(basePath + suffix + extPath, "GIF")

def savePictureWithPoints(filename, projectionMatrix, homopoints):
    projectedHomopoints = [ projectionMatrix.dot(p) for p in homopoints ]
    projectedPoints = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in projectedHomopoints ]
    drawPoints(filename, projectedPoints)

# normalized coordinates: [RC] * x
# http://wiki.icub.org/wiki/Image_Coordinate_Standard
def normalizedPoint(p, imageSize):
    (w, h) = imageSize
    (u, v) = p
    return (2*u / w - 1, 2*v / h - 1)

def denormalizedPoint(p, imageSize):
    (w, h) = imageSize
    (x, y) = p
    return (w*(x+1) / 2, h*(y+1) / 2)

def correctedPoint(p, distortion):
    k1, k2 = distortion
    r2 = p[0]*p[0] + p[1]*p[1]
    # coefficient = 1 + k1*r2 + k2*r2*r2
    # return (p[0]*coefficient, p[1]*coefficient)
    return (p[0] + p[0] * k1 * r2 + p[0] * k2 * r2 * r2, p[1] + p[1] * k1 * r2 + p[1] * k2 * r2 * r2)

def savePictureWithCorrectedPoints(filename, intrinsicMtx, extrinsicMatrix, homopoints, distortion):
    imageSize = (640, 480)
    # projectionMatrix = intrinsicMtx.dot(extrinsicMatrix)
    points1 = [ extrinsicMatrix.dot(p) for p in homopoints ]
    projectedPoints = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in points1 ]
    correctedPoints = [ correctedPoint(p, distortion) for p in projectedPoints ]
    homopoints2 = [ (p[0], p[1], 1.0) for p in correctedPoints ]
    points3 = [ intrinsicMtx.dot(p) for p in homopoints2 ]
    points4 = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in points3 ]
    # doubleHomopoints = [np.longdouble(p) for p in homopoints]
    # projectedHomopoints = [ projectionMatrix.dot(p) for p in doubleHomopoints ]
    # projectedPoints = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in projectedHomopoints ]
    # normalizedPoints = [ normalizedPoint(p, imageSize) for p in projectedPoints ]
    # correctedPoints = [ correctedPoint(p, distortion) for p in normalizedPoints ]
    # denormalizedPoints = [ denormalizedPoint(p, imageSize) for p in correctedPoints ]
    drawPoints(filename, points4, suffix = "-correctedpoints")

def run():
    distortion, intrinsicMtx, mtxs = loadCalibData("data/task34/Calib.txt")
    points = loadModelPoints("data/task34/Model.txt")
    homopoints = [ (p[0], p[1], 0.0, 1.0) for p in points ]

    # Undistorted pictures
    savePictureWithPoints("data/task34/UndistortIm1.gif", mtxs[0], homopoints)
    savePictureWithPoints("data/task34/UndistortIm2.gif", mtxs[1], homopoints)

    # Distorted pictures
    savePictureWithPoints("data/task34/CalibIm1.gif", mtxs[0], homopoints)
    savePictureWithPoints("data/task34/CalibIm2.gif", mtxs[1], homopoints)
    savePictureWithPoints("data/task34/CalibIm3.gif", mtxs[2], homopoints)
    savePictureWithPoints("data/task34/CalibIm4.gif", mtxs[3], homopoints)
    savePictureWithPoints("data/task34/CalibIm5.gif", mtxs[4], homopoints)

    savePictureWithCorrectedPoints("data/task34/CalibIm1.gif", intrinsicMtx, mtxs[0], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm2.gif", intrinsicMtx, mtxs[1], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm3.gif", intrinsicMtx, mtxs[2], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm4.gif", intrinsicMtx, mtxs[3], homopoints, distortion)
    savePictureWithCorrectedPoints("data/task34/CalibIm5.gif", intrinsicMtx, mtxs[4], homopoints, distortion)

run()
