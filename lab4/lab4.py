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
TRESHOLD = 1000
SEARCH_SIZE = 5

# Questions:
# Why are we doing the second gaussian? We we do not need to calculate a sum per each pixel point?

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

def anms_filter(cornersPositions, result):
    maximas = []
    cornerValues = result[tuple(cornersPositions.T)]
    corners = np.hstack((corners, cornerValues[:,None]))
    # Potem mozna te liste posortowac i wiemy ze po prostu musimy brac ja cala od gory
    # for (y, x) in corners:


    return np.array(maximas)


def harrisCornerDetector(filename, gaussianDerivativeSigma = 3.0, gaussianFilterSigma = 3.0):
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
    # for y in range(WINDOW_SIZE, image.shape[0] - WINDOW_SIZE):
    #     print(y)
    #     for x in range(WINDOW_SIZE, image.shape[1] - WINDOW_SIZE):
    #         # result[y][x] = computeResponseValue((x, y), ixix, ixiy, iyiy)
    #         result[y][x] = gixix[y][x]*giyiy[y][x] - gixiy[y][x]*gixiy[y][x] - ALPHA * np.power(gixix[y][x] + giyiy[y][x], 2)
    result = gixix*giyiy - gixiy*gixiy - ALPHA * np.power(gixix + giyiy, 2)

    heatMap = np.log(np.where(result < 1, np.ones(result.shape), result))
    sc.misc.imsave(mkPath(filename, "-4-heat-map"), heatMap)

    print("Finding maximas...")
    corners = np.argwhere(result > TRESHOLD)
    maximaCorners = filter_maxima(corners, result)
    print("Filtered from %d to %d points" % (len(corners), len(maximaCorners)))

    maximaCornersAnms = anms_filter(corners, result)
    print("Filtered from %d to %d points" % (len(corners), len(maximaCornersAnms)))

    drawCorners(filename, maximaCorners, "-5-with-corners")
    drawCorners(filename, maximaCornersAnms, "-5-with-corners-anms")


def run():
    filenames = [
            "data/Notre Dame/1_o.jpg",
            "data/Notre Dame/2_o.jpg",
            "data/Mount Rushmore/9021235130_7c2acd9554_o.jpg",
            "data/Mount Rushmore/9318872612_a255c874fb_o.jpg",
            "data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg",
            "data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg",
        ]
    for filename in filenames:
        print("=== File: %s" % filename)
        harrisCornerDetector(filename)

# run()
