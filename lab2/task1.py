import scipy as sc
import scipy.io
import numpy as np
import PIL as pil
import os.path
import Image, ImageDraw

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

def corrected_f(f):
    [l, s, r] = np.linalg.svd(f)
    s[2] = 0.0
    cf = l.dot(np.diag(s)).dot(r)
    return cf

def correctEssential(e):
    [l, s, r] = np.linalg.svd(e)
    avg = (s[0] + s[1]) / 2
    ce = l.dot(np.diag([avg, avg, 0.0])).dot(r)
    return ce

def getNormalizationMtx(imageSize):
    (w, h) = imageSize
    normalizationMtx = np.array([
        [ 2.0 / w, 0.0, -1.0 ],
        [ 0.0, 2.0 / h, -1.0 ],
        [ 0.0, 0.0, 1.0 ]
    ])
    return normalizationMtx

def calculateFundamentalMtx(points1, points2):
    points = np.hstack((points1, points2))[0:8]
    # Czy to dobra kolejnosc ?
    a = np.array([ (x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1) for (x1, y1, w1, x2, y2, w2) in points ])
    [l, s, r] = np.linalg.svd(a)
    f = r[-1].reshape(3,3)
    return f

def getImageSize(filename):
    im = Image.open(filename)
    return im.size

def rescale(f):
    return (f / f[2,2])

# Should draw on the second image too

# When drawing, do we convert from Homogenous coordinates???

# Why we adjust Essential matrix in different way, than fundamental matrix?

def run():
    mat = sc.io.loadmat("data/compEx1data.mat")['x']
    xim1 = mat[0,0].T
    xim2 = mat[1,0].T

    # "Random"
    pts_indices = [3, 42, 123, 234, 456, 567, 678, 890, 1024]
    pts_indices = [3, 20, 42, 64, 123, 234, 256, 456, 567, 600, 678, 890, 930, 1024]
    # my_points = xims[0:8]

    ### Task 1, 2
    # Drawing lines for uncorrected fundamental matrix
    points1 = xim1[pts_indices]
    points2 = xim2[pts_indices]
    fundamentalMtx = rescale(calculateFundamentalMtx(points1, points2))
    drawEpilines("data/kronan1.JPG", fundamentalMtx, points1, points2, "-phase1")
    drawEpilines("data/kronan2.JPG", fundamentalMtx, points2, points1, "-phase1")

    # Drawing lines for corrected fundamental matrix
    correctedFundamentalMtx = rescale(corrected_f(fundamentalMtx))
    drawEpilines("data/kronan1.JPG", correctedFundamentalMtx, points1, points2, "-phase2")
    drawEpilines("data/kronan2.JPG", correctedFundamentalMtx, points2, points1, "-phase2")

    ### Task 3
    imageSize1 = getImageSize("data/kronan1.JPG")
    imageSize2 = getImageSize("data/kronan2.JPG")

    normMtx1 = getNormalizationMtx(imageSize1)
    normMtx2 = getNormalizationMtx(imageSize2)

    normalizedPoints1 = normMtx1.dot(points1.T).T
    normalizedPoints2 = normMtx2.dot(points2.T).T

    normalizedFundamentalMtx = corrected_f(calculateFundamentalMtx(normalizedPoints1, normalizedPoints2))
    denormalizedFundamentalMtx = rescale(normMtx2.T.dot(normalizedFundamentalMtx).dot(normMtx1))

    drawEpilines("data/kronan1.JPG", denormalizedFundamentalMtx, points1, points2, "-phase3")
    drawEpilines("data/kronan2.JPG", denormalizedFundamentalMtx, points2, points1, "-phase3")

    ### Task 4
    intrinsicMtx = sc.io.loadmat("data/compEx3data.mat")['K']
    invIntrinsicMtx = np.linalg.inv(intrinsicMtx)

    normalizedPoints1 = invIntrinsicMtx.dot(points1.T).T
    normalizedPoints2 = invIntrinsicMtx.dot(points2.T).T

    essentialMtx = correctEssential(calculateFundamentalMtx(normalizedPoints1, normalizedPoints2))
    fundamentalMtxFromEssential = invIntrinsicMtx.T.dot(essentialMtx).dot(invIntrinsicMtx)

    drawEpilines("data/kronan1.JPG", rescale(fundamentalMtxFromEssential), points1, points2, "-phase4")
    drawEpilines("data/kronan2.JPG", rescale(fundamentalMtxFromEssential), points2, points1, "-phase4")

run()
