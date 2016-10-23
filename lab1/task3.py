import numpy as np
import PIL as pil
import Image, ImageDraw
import os.path

def loadCalibrationMatrix(f):
    rows = []
    for i in range(0,3): # 4 lines
        line = f.readline().split()
        rows.append([float(i) for i in line])
    last_line = f.readline().split()
    last_column = np.array([[float(i)] for i in last_line])
    result = np.array(rows)
    return np.concatenate((result, last_column), axis=1)

def loadCalibData(filename):
    f = open(filename,'r')
    [a,b,c,u0,v0] = f.readline().split()
    intrinsicMtx = np.array([
        [float(a), float(b), float(u0)],
        [0.0, float(c), float(v0)],
        [0.0, 0.0, 1.0]
        ])
    f.readline()
    f.readline()
    f.readline()

    mtx1 = loadCalibrationMatrix(f)
    f.readline()
    mtx2 = loadCalibrationMatrix(f)
    f.readline()
    mtx3 = loadCalibrationMatrix(f)
    f.readline()
    mtx4 = loadCalibrationMatrix(f)
    f.readline()
    mtx5 = loadCalibrationMatrix(f)
    f.readline()

    return [intrinsicMtx, mtx1, mtx2, mtx3, mtx4, mtx5]

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

def drawPoints(filename, points):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    drawSmallCircle(draw, 100, 100)
    for point in points:
        drawSmallCircle(draw, point[0], point[1])
    basePath, extPath = os.path.splitext(filename)
    im.show(draw)
    # im.save(basePath + "-withpoints" + extPath, "GIF")

def run():
    [intr, mtx1, mtx2, mtx3, mtx4, mtx5] = task3.loadCalibData("data/task34/Calib.txt")
    points = task3.loadModelPoints("data/task34/Model.txt")
    homopoints = [ (p[0], p[1], 0.0, 1.0) for p in points ]
    p1 = intr.dot(mtx1)
    homopoints1 = [ p1.dot(p) for p in homopoints ]
    points1 = [ (hp[0] / hp[2], hp[1] / hp[2]) for hp in homopoints1 ]
    drawPoints("")
