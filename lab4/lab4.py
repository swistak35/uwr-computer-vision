import scipy as sc
import scipy.io
import scipy.misc
import scipy.ndimage
import numpy as np
import numpy.linalg
import PIL as pil
import os.path
import Image, ImageDraw
import time

ALPHA = 0.04
WINDOW_SIZE = 3 # window will be windowSize*2 + 1
TRESHOLD = 100
ANMS_COEFFICIENT = 0.98

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

def drawCornersWithRadius(filename, points, suffix):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for (x,y,r) in points:
        drawSmallCircle(draw, y, x, width = r)
        drawSmallCircle(draw, y, x, outlineColor = (255, 0, 0), width = 3)
    im.save(mkPath(filename, suffix), "JPEG")

def filter_maxima(corners, result, span = 5, coefficient = 1.0):
    maximas = []
    for (y, x) in corners:
        v = result[y,x] * coefficient
        top = np.max([y - span, 0])
        bottom = np.min([y + span + 1, result.shape[0] - 1])
        left = np.max([x - span, 0])
        right = np.min([x + span + 1, result.shape[1] - 1])
        window = result[top:bottom,left:right]
        if np.sum(window >= v) <= 1:
            maximas.append((y,x))
    return np.array(maximas)

def anms_filter(unfilteredCornersPositions, result):
    maximas = []
    cornersPositions = filter_maxima(unfilteredCornersPositions, result, span = 1, coefficient = 0.98)
    print("Initial pre-ANMS filtered from %d to %d points" % (len(unfilteredCornersPositions), len(cornersPositions)))
    cornerValues = result[tuple(cornersPositions.T)]
    corners = np.hstack((cornersPositions, cornerValues[:,None]))
    cornersSorted = corners[corners[:,2].argsort()][::-1]
    cornersRadiuses = []
    for (index, p) in enumerate(cornersSorted):
        v = p[2] * ANMS_COEFFICIENT
        bigger = cornersSorted[cornersSorted[:,2] > v]
        if bigger.shape[0] > 1:
            r = np.partition(np.linalg.norm(bigger[:,0:2] - p[0:2], axis = 1), 1)[1]
            cornersRadiuses.append(r)
        else:
            cornersRadiuses.append(np.inf)
    cornersRadiuses = np.array(cornersRadiuses)
    cornersWithRadiuses = np.hstack((cornersSorted, cornersRadiuses[:,None]))
    cornersRadiusesSorted = cornersWithRadiuses[cornersWithRadiuses[:,3].argsort()][::-1]
    # print(cornersRadiusesSorted[0:20])
    # print(np.sum(cornersRadiusesSorted[:,3] > 1.44))
    return cornersRadiusesSorted[0:100,(0,1,3)]


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
    result = gixix*giyiy - gixiy*gixiy - ALPHA * np.power(gixix + giyiy, 2)

    heatMap = np.log(np.where(result < 1, np.ones(result.shape), result))
    sc.misc.imsave(mkPath(filename, "-4-heat-map"), heatMap)

    print("Finding maximas...")
    cornersPositions = np.argwhere(result > TRESHOLD)
    maximaCorners = filter_maxima(cornersPositions, result, span = 5)
    print("Filtered from %d to %d points" % (len(cornersPositions), len(maximaCorners)))

    t1 = time.time()
    maximaCornersAnms = anms_filter(cornersPositions, result)
    t2 = time.time()
    print("Filtered from %d to %d points in %f seconds" % (len(cornersPositions), len(maximaCornersAnms), (t2 - t1)))

    drawCorners(filename, maximaCorners, "-5-with-corners")
    drawCornersWithRadius(filename, maximaCornersAnms, "-5-with-corners-anms")

def mkCurrentWindow(currentImage, y, x):
    return np.array([
            currentImage[y-1, x],
            currentImage[y+1, x],
            currentImage[y, x+1],
            currentImage[y, x-1],
            currentImage[y-1, x-1],
            currentImage[y+1, x-1],
            currentImage[y-1, x+1],
            currentImage[y+1, x+1],
        ])

def siftCornerDetector(filename):
    image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale

    # sigma = 1.6
    # k = 1.41
    # for i in range(10):
    #     print("Computing gaussian %d" % i)
    #     imageGaussian = sc.ndimage.filters.gaussian_filter(image, sigma*np.power(k, i), order = 2)
    #     sc.misc.imsave(mkPath(filename, "-1-sift-scale-%d" % i), imageGaussian)
    # for i in range(4):
    #     print("Computing zoom %d" % i)
    #     zoomed = sc.ndimage.zoom(image, np.power(0.5, i))
    #     sc.misc.imsave(mkPath(filename, "-1-zoom-%d" % i), zoomed)

    octaves = 2 # Should be 4 in final
    scalesPerOctave = 3
    sigma = 1.6
    k = 1.41
    allImages = []
    allDiffImages = []
    for octave in range(octaves):
        gaussianImages = []
        diffImages = []
        for scale in range(scalesPerOctave + 3):
            print("Computing octave %d scale %d" % (octave, scale))
            zoomedImage = sc.ndimage.zoom(image, np.power(0.5, octave))
            gaussianImage = sc.ndimage.filters.gaussian_filter(zoomedImage, sigma * np.power(k, scale), order = 2)
            sc.misc.imsave(mkPath(filename, "-1-sift-scale-%d-%d" % (octave, scale)), gaussianImage)
            if scale > 0:
                diffGaussian = gaussianImage - gaussianImages[-1]
                sc.misc.imsave(mkPath(filename, "-2-sift-diff-%d-%d" % (octave, scale)), diffGaussian)
                diffImages.append(diffGaussian)
            gaussianImages.append(gaussianImage)
        allImages.append(gaussianImages)
        allDiffImages.append(diffImages)

    previousfeatures = 0
    features = []
    for octave in range(octaves):
        for scale in range(1, scalesPerOctave + 1):
            print("Searching octave %d scale %d" % (octave, scale))
            belowImage = allDiffImages[octave][scale - 1]
            currentImage = allDiffImages[octave][scale]
            aboveImage = allDiffImages[octave][scale + 1]
            t1 = time.time()
            for y in range(1, currentImage.shape[0] - 2):
                for x in range(1, currentImage.shape[1] - 2):
                    belowWindow = belowImage[y-1:y+2,x-1:x+2]
                    assert(belowWindow.shape == (3,3))
                    aboveWindow = aboveImage[y-1:y+2,x-1:x+2]
                    assert(aboveWindow.shape == (3,3))
                    currentWindow = mkCurrentWindow(currentImage, y, x)
                    if np.all(belowWindow > currentImage[y,x]) and np.all(aboveWindow > currentImage[y,x]) and np.all(currentWindow > currentImage[y,x]):
                        features.append((y, x, octave, scale))
                    # if np.all(belowWindow < currentImage[y,x]) and np.all(aboveWindow < currentImage[y,x]) and np.all(currentWindow < currentImage[y,x]):
                        # features.append((y, x, octave, scale))
            t2 = time.time()
            print("Finished in %f" % (t2 - t1))
            print("Found %d new features" % (len(features) - previousfeatures))
            previousfeatures = len(features)
    print("Found %d features" % len(features))





def run():
    filenames = [
            "data/Notre Dame/1_o.jpg",
            # "data/Notre Dame/2_o.jpg",
            # "data/Mount Rushmore/9021235130_7c2acd9554_o.jpg",
            # "data/Mount Rushmore/9318872612_a255c874fb_o.jpg",
            # "data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg",
            # "data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg",
        ]
    for filename in filenames:
        print("=== File: %s" % filename)
        # harrisCornerDetector(filename)
        siftCornerDetector(filename)

run()
