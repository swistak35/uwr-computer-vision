import scipy as sc
import scipy.io
import numpy as np
import PIL as pil
import os.path
import Image, ImageDraw


AMOUNT_OF_POINTS_TO_DRAW = 15
AMOUNT_OF_POINTS_TO_ESTIMATE = 15

assert(AMOUNT_OF_POINTS_TO_DRAW >= AMOUNT_OF_POINTS_TO_ESTIMATE)

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

# Draw small circle on an image
# `draw` is a ImageDraw from PIL
def drawSmallCircle(draw, x, y, colorCircle = (0, 255, 0), width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = colorCircle)

def drawEpilines(filename, fundamentalMtx, points1, points2, suffix):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    i = 0
    for (p1, p2) in zip(points1, points2):
        lineColor = (200, 0, 240) if i < AMOUNT_OF_POINTS_TO_ESTIMATE else (200, 240, 0)
        circleColor = (0, 0, 240) if i < AMOUNT_OF_POINTS_TO_ESTIMATE else (0, 240, 0)
        l = fundamentalMtx.dot(p2)
        x1 = 0
        y1 = (-l[2] - l[0] * x1) / l[1]
        x2 = im.size[0]
        y2 = (-l[2] - l[0] * x2) / l[1]
        draw.line([(x1, y1), (x2, y2)], fill = lineColor, width = 3)
        drawSmallCircle(draw, p1[0], p1[1], colorCircle = circleColor, width = 4)
        i += 1
    im.save(mkPath(filename, suffix), "JPEG")

def correctFundamental(f):
    [l, s, r] = np.linalg.svd(f)
    s[2] = 0.0
    cf = l.dot(np.diag(s)).dot(r)
    return cf

def correctEssential(e):
    [l, s, r] = np.linalg.svd(e)
    avg = (s[0] + s[1]) / 2
    ce = l.dot(np.diag([avg, avg, 0.0])).dot(r)
    return ce

def getNormalizationMtx(points):
    meanX = np.mean(points[:,0])
    meanY = np.mean(points[:,1])
    stdX = np.std(points[:,0])
    stdY = np.std(points[:,1])
    mS = np.array([
        [ 1.41 / stdX, 0, 0 ],
        [ 0, 1.41 / stdY, 0],
        [0, 0, 1]])
    mT = np.array([ [1, 0, -meanX], [0, 1, -meanY], [0, 0, 1] ])
    r = mS.dot(mT)
    pointsd = r.dot(points.T).T
    avgDist = np.mean([ np.sqrt(x * x + y * y) / (w*w) for (x,y,w) in pointsd ])
    return r

def calculateFundamentalMtx(points1, points2):
    points = np.hstack((points1, points2))[0:AMOUNT_OF_POINTS_TO_ESTIMATE]
    # a = np.array([ (x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1) for (x1, y1, w1, x2, y2, w2) in points ])
    # a = np.array([ ( x * xp, y * xp, xp, x * yp, y * yp, yp, x, y, 1 ) for (x, y, w, xp, yp, wp) in points ])
    a = np.array([ (x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1) for (x2, y2, w2, x1, y1, w1) in points ])
    [l, s, r] = np.linalg.svd(a)
    f1 = r[-1].reshape(3,3)

    return f1

def rescale(f):
    # return f
    return (f / f[2,2])

def loadPointsData():
    mat = sc.io.loadmat("data/compEx1data.mat")['x']
    xim1 = mat[0,0].T
    xim2 = mat[1,0].T

    return (xim1, xim2)

def findRandomPoints(pointsData):
    xim1, xim2 = pointsData
    pts_indices = np.random.choice(xim1.shape[0], AMOUNT_OF_POINTS_TO_DRAW, replace=False)
    # "Random"
    pts_indices = [1592,  526,  393, 1403,  433,  576, 1087,  429,  611,  530,  812, 1449,  651,  561, 1018]

    return pts_indices

# Drawing lines for uncorrected fundamental matrix
def task1(pointsData, pts_indices):
    xim1, xim2 = pointsData
    points1 = xim1[pts_indices]
    points2 = xim2[pts_indices]
    fundamentalMtx = rescale(calculateFundamentalMtx(points1, points2))
    drawEpilines("data/kronan1.JPG", fundamentalMtx.T, points1, points2, "-phase1")
    drawEpilines("data/kronan2.JPG", fundamentalMtx, points2, points1, "-phase1")

# Drawing lines for corrected fundamental matrix
def task2(pointsData, pts_indices):
    xim1, xim2 = pointsData
    points1 = xim1[pts_indices]
    points2 = xim2[pts_indices]
    fundamentalMtx = rescale(calculateFundamentalMtx(points1, points2))
    correctedFundamentalMtx = rescale(correctFundamental(fundamentalMtx))
    drawEpilines("data/kronan1.JPG", correctedFundamentalMtx.T, points1, points2, "-phase2")
    drawEpilines("data/kronan2.JPG", correctedFundamentalMtx, points2, points1, "-phase2")

def task3(pointsData, pts_indices):
    xim1, xim2 = pointsData
    points1 = xim1[pts_indices]
    points2 = xim2[pts_indices]

    normMtx1 = getNormalizationMtx(xim1)
    normMtx2 = getNormalizationMtx(xim2)

    normalizedPoints1 = normMtx1.dot(points1.T).T
    normalizedPoints2 = normMtx2.dot(points2.T).T

    normalizedFundamentalMtx = rescale(correctFundamental(calculateFundamentalMtx(normalizedPoints1, normalizedPoints2)))
    denormalizedFundamentalMtx = rescale(normMtx2.T.dot(normalizedFundamentalMtx).dot(normMtx1))

    drawEpilines("data/kronan1.JPG", denormalizedFundamentalMtx.T, points1, points2, "-phase3")
    drawEpilines("data/kronan2.JPG", denormalizedFundamentalMtx, points2, points1, "-phase3")

def task4(pointsData, pts_indices):
    xim1, xim2 = pointsData
    points1 = xim1[pts_indices]
    points2 = xim2[pts_indices]

    intrinsicMtx = sc.io.loadmat("data/compEx3data.mat")['K']
    invIntrinsicMtx = np.linalg.inv(intrinsicMtx)

    normalizedPoints1 = invIntrinsicMtx.dot(points1.T).T
    normalizedPoints2 = invIntrinsicMtx.dot(points2.T).T

    essentialMtx = correctEssential(calculateFundamentalMtx(normalizedPoints1, normalizedPoints2))
    fundamentalMtxFromEssential = rescale(invIntrinsicMtx.T.dot(essentialMtx).dot(invIntrinsicMtx))

    drawEpilines("data/kronan1.JPG", fundamentalMtxFromEssential.T, points1, points2, "-phase4")
    drawEpilines("data/kronan2.JPG", fundamentalMtxFromEssential, points2, points1, "-phase4")

def run():
    pointsData = loadPointsData()
    pts_indices = findRandomPoints(pointsData)

    task1(pointsData, pts_indices)
    task2(pointsData, pts_indices)
    task3(pointsData, pts_indices)
    task4(pointsData, pts_indices)

    ### Task 5
    # P1 = np.hstack((np.diag((1,1,1)), np.zeros(3).reshape(3,1)))

run()
