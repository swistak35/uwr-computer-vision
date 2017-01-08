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

octaves = 4
scalesPerOctave = 3
sigma = 1.6
k = np.power(2, 1.0 / scalesPerOctave)
THRESHOLD = 0.03 * 255.0 # In paper they say: 0.03 is nice threshold, assuming values are from 0.0 to 1.0. Our format is up to 255.0?
EDGE_RATIO = 10.0
SHOULD_DRAW_REMOVED_EDGES = True
SHOULD_DRAW_REMOVED_LOWCONTRAST = False

COLORS = {
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
}

# How to draw the circle based on the sigma?
# TODO: Make constants capital letter
# TODO: octave is not needed in the list of features

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = COLORS['GREEN'], width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawFeature(draw, x, y, scale, octave, outlineColor = COLORS['GREEN']):
    realX = x * np.power(2, octave)
    realY = y * np.power(2, octave)
    radius = 5 + sigma * np.power(k, octave * scalesPerOctave + scale)
    drawSmallCircle(draw, realX, realY, width = radius, outlineColor = outlineColor)

def drawFeatures(filename, features, featuresBelowThreshold, edgesRemoved):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for featuresInOctave in features:
        for (y, x, octave, scale, v, ev) in featuresInOctave:
            drawFeature(draw, x, y, scale, octave)
    if SHOULD_DRAW_REMOVED_LOWCONTRAST:
        for featuresBelowThresholdInOctave in featuresBelowThreshold:
            for (y, x, octave, scale, v, ev) in featuresBelowThresholdInOctave:
                drawFeature(draw, x, y, scale, octave, outlineColor = COLORS['BLUE'])
    if SHOULD_DRAW_REMOVED_EDGES:
        for edgesRemovedInOctave in edgesRemoved:
            for (y, x, octave, scale, v, ev) in edgesRemovedInOctave:
                drawFeature(draw, x, y, scale, octave, outlineColor = COLORS['RED'])
    im.save(mkPath(filename, "-with-features"), "JPEG")

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

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

def buildMinMaxImages(allDiffImages):
    print("Building min/max arrays")
    minImages = []
    maxImages = []
    for octave in range(octaves):
        minImagesInOctave = []
        maxImagesInOctave = []
        for scale in range(scalesPerOctave + 2):
            img = allDiffImages[octave][scale]
            (height, width) = img.shape
            arrayCollection = np.array([
                    img[0:(height - 2),1:(width - 1)],
                    img[2:(height),1:(width - 1)],
                    img[1:(height - 1),0:(width - 2)],
                    img[1:(height - 1),2:(width)],

                    img[0:(height - 2),0:(width - 2)],
                    img[0:(height - 2),2:(width)],
                    img[2:(height),0:(width - 2)],
                    img[2:(height),2:(width)],
                ])
            mins = np.amin(arrayCollection, axis = 0)
            maxs = np.amax(arrayCollection, axis = 0)
            minImagesInOctave.append(mins)
            maxImagesInOctave.append(maxs)
        minImages.append(minImagesInOctave)
        maxImages.append(maxImagesInOctave)
    return (minImages, maxImages)

def featuresAmount(features):
    return np.sum([ len(features[o]) for o in range(octaves) ])

def findFeatures2(allDiffImages):
    print("== Searching for keypoints...")
    (minImages, maxImages) = buildMinMaxImages(allDiffImages)

    features = []
    for octave in range(octaves):
        featuresInOctave = []
        (height, width) = allDiffImages[octave][0].shape
        for scale in range(1, scalesPerOctave + 1):
            print("Searching octave %d scale %d" % (octave, scale))
            for y in range(1, height - 2):
                for x in range(1, width - 2):
                    belowImageV = allDiffImages[octave][scale - 1][y,x]
                    currentImageV = allDiffImages[octave][scale][y,x]
                    aboveImageV = allDiffImages[octave][scale + 1][y,x]
                    if minImages[octave][scale - 1][y-1,x-1] > currentImageV and minImages[octave][scale + 1][y-1,x-1] > currentImageV and minImages[octave][scale][y-1,x-1] > currentImageV and belowImageV > currentImageV and aboveImageV > currentImageV:
                        # TODO: Maybe we could here already add a value?
                        featuresInOctave.append((y, x, octave, scale))
                    if maxImages[octave][scale - 1][y-1,x-1] < currentImageV and maxImages[octave][scale + 1][y-1,x-1] < currentImageV and maxImages[octave][scale][y-1,x-1] < currentImageV and belowImageV < currentImageV and aboveImageV < currentImageV:
                        featuresInOctave.append((y, x, octave, scale))
        features.append(featuresInOctave)
    return features

# Fast way to compute hessian
# http://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
# In our case, result is:
# | ss | sy | sx |
# | ys | yy | yx |
# | xs | xy | xx |
def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

def computeExtremaValues(allDiffImages, features):
    print("Computing extrema values")
    featuresWithExtremas = []
    for octave in range(octaves):
        featuresInOctave = features[octave]
        # TODO: That could be done once, no need to make this array each time - or maybe it is even already done?
        diffsInOctave = allDiffImages[octave]
        fullHessian = hessian(diffsInOctave)

        featuresInOctaveWithExtremas = []
        for f in featuresInOctave:
            (y, x, octave, s) = f

            localHessian = fullHessian[:, :, s, y, x]
            invHessian = np.linalg.inv(localHessian)
            # TODO: This could be optimized using np.gradient
            dDX = np.array([
                    (diffsInOctave[s+1][y, x] - diffsInOctave[s-1][y, x]) / 2,
                    (diffsInOctave[s][y+1, x] - diffsInOctave[s][y-1, x]) / 2,
                    (diffsInOctave[s][y, x+1] - diffsInOctave[s][y, x-1]) / 2,
                ])
            hvec = -invHessian.dot(dDX)
            # This could be done in pure numpy?
            extremaValue = diffsInOctave[s][y, x] + dDX.dot(hvec) + 0.5*hvec.dot(localHessian).dot(hvec)
            featuresInOctaveWithExtremas.append(f + (diffsInOctave[s][y,x], extremaValue))
            # print("Normal value = %f \t Extrema value = %f \t Diff = %f" % (diffsInOctave[s][y,x], extremaValue, extremaValue - diffsInOctave[s][y,x]))
        featuresWithExtremas.append(np.array(featuresInOctaveWithExtremas))
    return featuresWithExtremas

def filterFeaturesAboveThreshold(features):
    print("Filtering features above threshold")
    featuresAboveThreshold = []
    featuresBelowThreshold = []
    for featuresInOctave in features:
        featuresAboveThreshold.append(featuresInOctave[np.abs(featuresInOctave[:,5] > THRESHOLD)])
        featuresBelowThreshold.append(featuresInOctave[np.abs(featuresInOctave[:,5] <= THRESHOLD)])
    return (featuresAboveThreshold, featuresBelowThreshold)

def filterEdgeFeatures(featuresAboveThreshold, allDiffImages):
    flattenedDiffs = np.array(allDiffImages[0])
    hessian2x2 = []
    for img in flattenedDiffs:
        hessian2x2.append(hessian(img))
    notEdges = []
    edgesRemoved = []
    for featuresInOctave in featuresAboveThreshold:
        notEdgesInOctave = []
        edgesRemovedInOctave = []
        for f in featuresInOctave:
            (y, x, octave, s, v, ev) = f
            if octave == 0:
                hs = hessian2x2[int(s)][:,:,int(y),int(x)] # How int rounds? Potential fuckup?
                r = np.power(np.trace(hs), 2) / np.linalg.det(hs)
                edge_ratio_coefficient = np.power(EDGE_RATIO + 1, 2) / EDGE_RATIO
                if r < edge_ratio_coefficient:
                    notEdgesInOctave.append(f)
                else:
                    edgesRemovedInOctave.append(f)
            else:
                notEdgesInOctave.append(f)
        notEdges.append(notEdgesInOctave)
        edgesRemoved.append(edgesRemovedInOctave)
    return (notEdges, edgesRemoved)

def filterFeatures(allDiffImages, features):
    print("== Filtering features...")

    featuresWithExtremas = computeExtremaValues(allDiffImages, features)

    (featuresAboveThreshold, featuresBelowThreshold) = filterFeaturesAboveThreshold(featuresWithExtremas)

    print("Filtering edges")
    (notEdges, edgesRemoved) = filterEdgeFeatures(featuresAboveThreshold, allDiffImages)

    return (notEdges, featuresBelowThreshold, edgesRemoved)

def findDiffImages(filename):
    image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale
    print("== Computing DoG images...")

    # This could be optimized, a lot of repetitive work here
    allImages = []
    allDiffImages = []
    for octave in range(octaves):
        gaussianImages = []
        diffImages = []
        for scale in range(scalesPerOctave + 3):
            print("Computing octave %d scale %d" % (octave, scale))
            zoomedImage = sc.ndimage.zoom(image, np.power(0.5, octave))
            gaussianImage = sc.ndimage.filters.gaussian_filter(zoomedImage, sigma * np.power(k, scale), order = 0)
            # sc.misc.imsave(mkPath(filename, "-1-sift-scale-%d-%d" % (octave, scale)), gaussianImage)
            if scale > 0:
                diffGaussian = gaussianImage - gaussianImages[-1]
                # sc.misc.imsave(mkPath(filename, "-2-sift-diff-%d-%d" % (octave, scale)), diffGaussian)
                diffImages.append(diffGaussian)
            gaussianImages.append(gaussianImage)
        allImages.append(gaussianImages)
        allDiffImages.append(np.array(diffImages))
    return allDiffImages

def siftCornerDetector(filename):
    allDiffImages = findDiffImages(filename)

    t1 = time.time()
    features = findFeatures2(allDiffImages)
    t2 = time.time()
    print("Found %d features in %f" % (featuresAmount(features), t2 - t1))

    t1 = time.time()
    (features2, featuresBelowThreshold, edgesRemoved) = filterFeatures(allDiffImages, features)
    t2 = time.time()
    print("Filtered to %d features in %f (%d low-contrast removed, %d edges removed)" % (featuresAmount(features2), t2 - t1, featuresAmount(featuresBelowThreshold), featuresAmount(edgesRemoved)))

    drawFeatures(filename, features2, featuresBelowThreshold, edgesRemoved)

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
        siftCornerDetector(filename)

run()
