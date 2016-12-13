import scipy as sc
import scipy.io
import scipy.misc
import scipy.ndimage
import numpy as np
import numpy.linalg
import PIL as pil
import os.path
import Image, ImageDraw

ALPHA = 0.05
WINDOW_SIZE = 3 # window will be windowSize*2 + 1
TRESHOLD = 10000
SEARCH_SIZE = 5

# Questions:
# How to set the threshold? Some value like 95% percentile?

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = (0, 255, 0), width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawCorners(filename, points, suffix):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for (x,y) in points:
        drawSmallCircle(draw, y, x, width = 6)
    im.save(mkPath(filename, suffix), "JPEG")

# def computeResponseValue(coords, ixix, ixiy, iyiy):
#     (x, y) = coords
#     mtx00 = np.sum(ixix[(y-WINDOW_SIZE):(y+WINDOW_SIZE+1),(x-WINDOW_SIZE):(x+WINDOW_SIZE+1)])
#     mtx01 = np.sum(ixiy[(y-WINDOW_SIZE):(y+WINDOW_SIZE+1),(x-WINDOW_SIZE):(x+WINDOW_SIZE+1)])
#     mtx11 = np.sum(iyiy[(y-WINDOW_SIZE):(y+WINDOW_SIZE+1),(x-WINDOW_SIZE):(x+WINDOW_SIZE+1)])
#     secondMomentMatrix = np.array([[ mtx00, mtx01 ], [ mtx01, mtx11 ]])
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
    # response2 = np.linalg.det(secondMomentMatrix) - ALPHA * np.power(np.trace(secondMomentMatrix), 2)
    # print("Responses:")
    # print([response1, response2])
    # return response2

def filter_maxima(corners, result):
    maximas = []
    for (y, x) in corners:
        top = np.max([y - SEARCH_SIZE, 0])
        bottom = np.min([y + SEARCH_SIZE + 1, result.shape[0] - 1])
        left = np.max([x - SEARCH_SIZE, 0])
        right = np.min([x + SEARCH_SIZE + 1, result.shape[1] - 1])
        window = result[top:bottom,left:right]
        if result[y,x] == np.max(window):
            maximas.append((y,x))
    return np.array(maximas)

def harrisCornerDetector(filename, gaussianDerivativeSigma = 1.0, gaussianFilterSigma = 1.0):
    image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale

    imageGaussian0 = sc.ndimage.filters.gaussian_filter1d(image, gaussianDerivativeSigma, order = 1, axis = 0)
    imageGaussian1 = sc.ndimage.filters.gaussian_filter1d(image, gaussianDerivativeSigma, order = 1, axis = 1)

    sc.misc.imsave(mkPath(filename, "-1x-gauss-derivative-0"), imageGaussian0)
    sc.misc.imsave(mkPath(filename, "-1y-gauss-derivative-1"), imageGaussian1)

    result = np.zeros(image.shape)

    ixix = imageGaussian0 * imageGaussian0
    ixiy = imageGaussian0 * imageGaussian1
    iyiy = imageGaussian1 * imageGaussian1

    sc.misc.imsave(mkPath(filename, "-2-ixix"), ixix)
    sc.misc.imsave(mkPath(filename, "-2-ixiy"), ixiy)
    sc.misc.imsave(mkPath(filename, "-2-iyiy"), iyiy)

    gixix = sc.ndimage.filters.gaussian_filter(ixix, sigma = gaussianFilterSigma)
    gixiy = sc.ndimage.filters.gaussian_filter(ixiy, sigma = gaussianFilterSigma)
    giyiy = sc.ndimage.filters.gaussian_filter(iyiy, sigma = gaussianFilterSigma)

    sc.misc.imsave(mkPath(filename, "-3-gixix"), gixix)
    sc.misc.imsave(mkPath(filename, "-3-gixiy"), gixiy)
    sc.misc.imsave(mkPath(filename, "-3-giyiy"), giyiy)

    print("=== Computing response values...")
    for y in range(WINDOW_SIZE, image.shape[0] - WINDOW_SIZE):
        print(y)
        for x in range(WINDOW_SIZE, image.shape[1] - WINDOW_SIZE):
            # result[y][x] = computeResponseValue((x, y), ixix, ixiy, iyiy)
            result[y][x] = gixix[y][x]*giyiy[y][x] - gixiy[y][x]*gixiy[y][x] - ALPHA * np.power(gixix[y][x] + giyiy[y][x], 2)

    heatMap = np.log(np.where(result < 1, np.ones(result.shape), result))
    sc.misc.imsave(mkPath(filename, "-4-heat-map"), heatMap)

    print("Finding maximas...")
    corners = np.argwhere(result > TRESHOLD)
    maximaCorners = filter_maxima(corners, result)
    print("Filtered from %d to %d points" % (len(corners), len(maximaCorners)))

    drawCorners(filename, maximaCorners, "-5-with-corners")


def run():
    filename = "data/Notre Dame/1_o.jpg"
    harrisCornerDetector(filename)

run()
