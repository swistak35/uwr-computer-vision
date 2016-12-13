import scipy as sc
import scipy.io
import scipy.misc
import scipy.ndimage
import numpy as np
import numpy.linalg
# import cv2
import PIL as pil
import os.path
import Image, ImageDraw

ALPHA = 0.05
WINDOW_SIZE = 3 # window will be windowSize*2 + 1
TRESHOLD = 10000

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

def drawSmallCircle(draw, x, y, colorCircle = (0, 255, 0), width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = colorCircle)

def drawCorners(filename, points, suffix):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for (x,y) in points:
        drawSmallCircle(draw, x, y, width = 4)
    # i = 0
    # for (p1, p2) in zip(points1, points2):
    #     lineColor = (200, 0, 240) if i < AMOUNT_OF_POINTS_TO_ESTIMATE else (200, 240, 0)
    #     circleColor = (0, 0, 240) if i < AMOUNT_OF_POINTS_TO_ESTIMATE else (0, 240, 0)
    #     l = fundamentalMtx.dot(p2)
    #     x1 = 0
    #     y1 = (-l[2] - l[0] * x1) / l[1]
    #     x2 = im.size[0]
    #     y2 = (-l[2] - l[0] * x2) / l[1]
    #     draw.line([(x1, y1), (x2, y2)], fill = lineColor, width = 3)
    #     drawSmallCircle(draw, p1[0], p1[1], colorCircle = circleColor, width = 4)
    #     i += 1
    im.save(mkPath(filename, suffix), "JPEG")

def computeResponseValue(coords, ixix, ixiy, iyiy):
    (x, y) = coords
    mtx00 = np.sum(ixix[(y-WINDOW_SIZE):(y+WINDOW_SIZE+1),(x-WINDOW_SIZE):(x+WINDOW_SIZE+1)])
    mtx01 = np.sum(ixiy[(y-WINDOW_SIZE):(y+WINDOW_SIZE+1),(x-WINDOW_SIZE):(x+WINDOW_SIZE+1)])
    mtx11 = np.sum(iyiy[(y-WINDOW_SIZE):(y+WINDOW_SIZE+1),(x-WINDOW_SIZE):(x+WINDOW_SIZE+1)])
    secondMomentMatrix = np.array([[ mtx00, mtx01 ], [ mtx01, mtx11 ]])
    # secondMomentMatrix = np.zeros(4).reshape(2,2)
    # for yd in range(-WINDOW_SIZE, WINDOW_SIZE+1):
    #     for xd in range(-WINDOW_SIZE, WINDOW_SIZE+1):
    #         ix = imgGauss0[y + yd][x + xd] # Are these used in correct order?
    #         iy = imgGauss1[y + yd][x + xd]
    #         secondMomentMatrix[0][0] += ix * ix
    #         secondMomentMatrix[0][1] += ix * iy
    #         secondMomentMatrix[1][0] += ix * iy
    #         secondMomentMatrix[1][1] += iy * iy
    # eigvals = np.linalg.eigvals(secondMomentMatrix)
    # print("Eigvals:")
    # print(eigvals)
    # response1 = eigvals[0] * eigvals[1] - ALPHA * np.power(eigvals[0] + eigvals[1], 2)
    response2 = np.linalg.det(secondMomentMatrix) - ALPHA * np.power(np.trace(secondMomentMatrix), 2)
    # print("Responses:")
    # print([response1, response2])
    return response2

filename = "data/Notre Dame/1_o.jpg"
image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale

imageGaussian0 = sc.ndimage.filters.gaussian_filter1d(image, 1.0, order = 1, axis = 0)
imageGaussian1 = sc.ndimage.filters.gaussian_filter1d(image, 1.0, order = 1, axis = 1)

sc.misc.imsave(mkPath(filename, "-gauss-derivative-0"), imageGaussian0)
sc.misc.imsave(mkPath(filename, "-gauss-derivative-1"), imageGaussian1)

result = np.zeros(image.shape)

ixix = imageGaussian0 * imageGaussian0
ixiy = imageGaussian0 * imageGaussian1
iyiy = imageGaussian1 * imageGaussian1

sc.misc.imsave(mkPath(filename, "-ixix"), ixix)
sc.misc.imsave(mkPath(filename, "-ixiy"), ixiy)
sc.misc.imsave(mkPath(filename, "-iyiy"), iyiy)

gixix = sc.ndimage.filters.gaussian_filter(ixix, sigma = 1.0)
gixiy = sc.ndimage.filters.gaussian_filter(ixiy, sigma = 1.0)
giyiy = sc.ndimage.filters.gaussian_filter(iyiy, sigma = 1.0)

sc.misc.imsave(mkPath(filename, "-gixix"), gixix)
sc.misc.imsave(mkPath(filename, "-gixiy"), gixiy)
sc.misc.imsave(mkPath(filename, "-giyiy"), giyiy)

for y in range(WINDOW_SIZE, image.shape[0] - WINDOW_SIZE):
    print(y)
    for x in range(WINDOW_SIZE, image.shape[1] - WINDOW_SIZE):
        # result[y][x] = computeResponseValue((x, y), ixix, ixiy, iyiy)
        result[y][x] = gixix[y][x]*giyiy[y][x] - gixiy[y][x]*gixiy[y][x] - ALPHA * np.power(gixix[y][x] + giyiy[y][x], 2)

def filter_maxima(corners):
    SEARCH_SIZE = 5
    fullsearchsize = np.power(SEARCH_SIZE * 2 + 1, 2)
    maximas = []
    for i in reversed(xrange(len(corners))):
        first = i - fullsearchsize
        first = 0 if first < 0 else first
        close_enough = []
        for p in corners[first:i]:
            if np.linalg.norm(p - corners[i]) < SEARCH_SIZE:
                close_enough.append(p)
        if close_enough == []:
            maximas.append(corners[i])
    return np.array(maximas)

print("Finding maximas...")
corners = np.argwhere(result > TRESHOLD)
maximaCorners = filter_maxima(corners)

drawCorners(filename, maximaCorners, "-with-corners")

maximumValue = np.amax(result)
sc.misc.imsave(mkPath(filename, "-heat-map1"), result)

result = result / maximumValue
sc.misc.imsave(mkPath(filename, "-heat-map2"), result)
