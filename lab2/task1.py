import scipy as sc
import scipy.io
import numpy as np
import PIL as pil
import os.path
import Image, ImageDraw

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

def drawEpilines(filename, fundamentalMtx, points, suffix):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for mp in points:
        l = fundamentalMtx.dot(mp)
        x1 = 0
        y1 = (-l[2] - l[0] * x1) / l[1]
        x2 = im.size[0]
        y2 = (-l[2] - l[0] * x2) / l[1]
        draw.line([(x1, y1), (x2, y2)], fill = 200, width = 3)
    im.save(mkPath(filename, suffix), "JPEG")

def corrected_f(f):
    [l, s, r] = np.linalg.svd(f)
    s[2] = 0.0
    cf = l.dot(np.diag(s)).dot(r)
    return cf

def getNormalizationMtx(imageSize):
    (w, h) = imageSize
    normalizationMtx = np.array([
        [ 2.0 / w, 0.0, -1.0 ],
        [ 0.0, 2.0 / h, -1.0 ],
        [ 0.0, 0.0, 1.0 ]
    ])
    return normalizationMtx

def calculateFundamentalMtx(points1, points2):
    points = np.hstack((points1, points2))
    a = np.array([ (x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1) for (x1, y1, w1, x2, y2, w2) in points ])
    [l, s, r] = np.linalg.svd(a)
    f = r[-1].reshape(3,3)
    return f

def getImageSize(filename):
    im = Image.open(filename)
    return im.size

mat = sc.io.loadmat("data/compEx1data.mat")['x']
xim1 = mat[0,0].T
xim2 = mat[1,0].T
pts_indices = [3, 42, 123, 234, 456, 567, 678, 890]
# my_points = xims[0:8]
ps1 = xim1[pts_indices]
ps2 = xim2[pts_indices]

f = calculateFundamentalMtx(ps1, ps2)
drawEpilines("data/kronan1.JPG", f, ps2, "-withepilines")

cf = corrected_f(f)
drawEpilines("data/kronan1.JPG", cf, ps2, "-withcorrectedepilines")

imageSize1 = getImageSize("data/kronan1.JPG")
imageSize2 = getImageSize("data/kronan2.JPG")

normMtx1 = getNormalizationMtx(imageSize1)
normMtx2 = getNormalizationMtx(imageSize2)

nps1 = normMtx1.dot(ps1.T).T
nps2 = normMtx2.dot(ps2.T).T

nf = corrected_f(calculateFundamentalMtx(nps1, nps2))
dnf = normMtx2.T.dot(nf).dot(normMtx1)

drawEpilines("data/kronan1.JPG", dnf, ps2, "-with_normalized_and_corrected_epilines")
