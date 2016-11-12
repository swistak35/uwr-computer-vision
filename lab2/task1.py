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
    for (p1, p2) in zip(points1, points2):
        l = fundamentalMtx.dot(p2)
        x1 = 0
        y1 = (-l[2] - l[0] * x1) / l[1]
        x2 = im.size[0]
        y2 = (-l[2] - l[0] * x2) / l[1]
        draw.line([(x1, y1), (x2, y2)], fill = 200, width = 3)
        drawSmallCircle(draw, p1[0], p1[1], width = 4)
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
    print avgDist
    return r

def calculateFundamentalMtx(points1, points2):
    points = np.hstack((points1, points2))[0:AMOUNT_OF_POINTS_TO_ESTIMATE]
    # Czy to dobra kolejnosc ?
    a = np.array([ (x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1) for (x1, y1, w1, x2, y2, w2) in points ])
    [l, s, r] = np.linalg.svd(a)
    f = r[-1].reshape(3,3)
    return f

def rescale(f):
    return f
    # return (f / f[2,2])

# Why we adjust Essential matrix in different way, than fundamental matrix?

def run():
    mat = sc.io.loadmat("data/compEx1data.mat")['x']
    xim1 = mat[0,0].T
    xim2 = mat[1,0].T

    # "Random"
    # pts_indices = [3, 42, 123, 234, 456, 567, 678, 890, 1024]
    # pts_indices = [3, 20, 42, 64, 123, 234, 256, 456, 567, 600, 678, 890, 930, 1024]

    # quite good on 1,2,4: [1190  138 1562  631  407 1790  791 1448  112 1482  185  251 1233 1096  793 1680 1149 1809 1294 1715  906 1533 1949  809  629 1806 1353 1092  932  333 775  998 1870 1823  106  990  730 1275 1449 1424]
    # pts_indices = np.random.choice(xim1.shape[0], AMOUNT_OF_POINTS_TO_DRAW)
    pts_indices = [ 262,  639, 1955, 1375,  558, 1464,   10, 1670, 1897,  731, 1937, 1504, 1748,  120,  906]
    print pts_indices

    ### Task 1, 2
    # Drawing lines for uncorrected fundamental matrix
    points1 = xim1[pts_indices]
    points2 = xim2[pts_indices]
    fundamentalMtx = rescale(calculateFundamentalMtx(points1, points2))
    drawEpilines("data/kronan1.JPG", fundamentalMtx, points1, points2, "-phase1")
    drawEpilines("data/kronan2.JPG", fundamentalMtx.T, points2, points1, "-phase1")

    # Drawing lines for corrected fundamental matrix
    correctedFundamentalMtx = rescale(correctFundamental(fundamentalMtx))
    drawEpilines("data/kronan1.JPG", correctedFundamentalMtx, points1, points2, "-phase2")
    drawEpilines("data/kronan2.JPG", correctedFundamentalMtx.T, points2, points1, "-phase2")

    ### Task 3
    normMtx1 = getNormalizationMtx(points1)
    normMtx2 = getNormalizationMtx(points2)

    normalizedPoints1 = normMtx1.dot(points1.T).T
    normalizedPoints2 = normMtx2.dot(points2.T).T

    normalizedFundamentalMtx = calculateFundamentalMtx(normalizedPoints1, normalizedPoints2)
    denormalizedFundamentalMtx = rescale(correctFundamental(normMtx2.T.dot(normalizedFundamentalMtx).dot(normMtx1)))

    drawEpilines("data/kronan1.JPG", denormalizedFundamentalMtx, points1, points2, "-phase3")
    drawEpilines("data/kronan2.JPG", denormalizedFundamentalMtx.T, points2, points1, "-phase3")

    ### Task 4
    intrinsicMtx = sc.io.loadmat("data/compEx3data.mat")['K']
    invIntrinsicMtx = np.linalg.inv(intrinsicMtx)

    normalizedPoints1 = invIntrinsicMtx.dot(points1.T).T
    normalizedPoints2 = invIntrinsicMtx.dot(points2.T).T

    essentialMtx = correctEssential(calculateFundamentalMtx(normalizedPoints1, normalizedPoints2))
    fundamentalMtxFromEssential = rescale(invIntrinsicMtx.T.dot(essentialMtx).dot(invIntrinsicMtx))

    drawEpilines("data/kronan1.JPG", fundamentalMtxFromEssential, points1, points2, "-phase4")
    drawEpilines("data/kronan2.JPG", fundamentalMtxFromEssential.T, points2, points1, "-phase4")

run()
