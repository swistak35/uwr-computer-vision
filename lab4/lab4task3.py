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

octaves = 4 # Should be 4 in final
scalesPerOctave = 3
sigma = 1.6
k = np.power(2, 1.0 / scalesPerOctave)
THRESHOLD = 0.03 * 255.0 # In paper they say: 0.03 is nice threshold, assuming values are from 0.0 to 1.0. Our format is up to 255.0?
EDGE_RATIO = 10.0

COLORS = {
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
}

# How to draw the circle based on the sigma?
# TODO: Draw the filtered features

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = COLORS['GREEN'], width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawFeatures(filename, features, edgesRemoved):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for featuresInOctave in features:
        for (y, x, octave, scale, v, ev) in featuresInOctave:
            realX = x * np.power(2, octave)
            realY = y * np.power(2, octave)
            # ...
            radius = 5 + sigma * np.power(k, octave * scalesPerOctave + scale)
            drawSmallCircle(draw, realX, realY, width = radius)
    for edgesRemovedInOctave in edgesRemoved:
        for (y, x, octave, scale, v, ev) in edgesRemovedInOctave:
            realX = x * np.power(2, octave)
            realY = y * np.power(2, octave)
            # ...
            radius = 5 + sigma * np.power(k, octave * scalesPerOctave + scale)
            drawSmallCircle(draw, realX, realY, width = radius, outlineColor = (255, 0, 0))
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

def findFeatures1(allDiffImages):
    features = []
    for octave in range(octaves):
        for scale in range(1, scalesPerOctave + 1):
            print("Searching octave %d scale %d" % (octave, scale))
            belowImage = allDiffImages[octave][scale - 1]
            currentImage = allDiffImages[octave][scale]
            aboveImage = allDiffImages[octave][scale + 1]
            for y in range(1, currentImage.shape[0] - 2):
                for x in range(1, currentImage.shape[1] - 2):
                    belowWindow = belowImage[y-1:y+2,x-1:x+2]
                    assert(belowWindow.shape == (3,3))
                    aboveWindow = aboveImage[y-1:y+2,x-1:x+2]
                    assert(aboveWindow.shape == (3,3))
                    currentWindow = mkCurrentWindow(currentImage, y, x)
                    if np.all(belowWindow > currentImage[y,x]) and np.all(aboveWindow > currentImage[y,x]) and np.all(currentWindow > currentImage[y,x]):
                        features.append((y, x, octave, scale))
                    if np.all(belowWindow < currentImage[y,x]) and np.all(aboveWindow < currentImage[y,x]) and np.all(currentWindow < currentImage[y,x]):
                        features.append((y, x, octave, scale))
    return features

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

# Not used in the end, only used as helper in debugging
# def dxy(images, x, y, s):
#     d1 = (images[s][y+1, x+1] - images[s][y+1,x-1]) / 2
#     d2 = (images[s][y-1, x+1] - images[s][y-1,x-1]) / 2
#     return (d1 - d2) / 2

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
    flattenedDiffs = np.array(allDiffImages[0])
    fullHessian = hessian(flattenedDiffs)
    for featuresInOctave in features:
        featuresInOctaveWithExtremas = []
        for f in featuresInOctave:
            (y, x, octave, s) = f
            if octave == 0:
                tmph = fullHessian[:, :, s, y, x]
                realHessian = np.array([
                        [ tmph[2, 2], tmph[2, 1], tmph[2, 0] ],
                        [ tmph[1, 2], tmph[1, 1], tmph[1, 0] ],
                        [ tmph[0, 2], tmph[0, 1], tmph[0, 0] ],
                    ])
                invHessian = np.linalg.inv(tmph)
                xvec = np.array([ x, y, s ])
                dDX = np.array([
                        (flattenedDiffs[s+1][y, x] - flattenedDiffs[s-1][y, x]) / 2,
                        (flattenedDiffs[s][y+1, x] - flattenedDiffs[s][y-1, x]) / 2,
                        (flattenedDiffs[s][y, x+1] - flattenedDiffs[s][y, x-1]) / 2,
                    ])
                hvec = -invHessian.dot(dDX)
                extremaValue = flattenedDiffs[s][y, x] + dDX.dot(hvec) + 0.5*hvec.dot(tmph).dot(hvec)
                # print("Normal value = %f \t Extrema value = %f \t Diff = %f" % (flattenedDiffs[s][y,x], extremaValue, extremaValue - flattenedDiffs[s][y,x]))
                featuresInOctaveWithExtremas.append(f + (flattenedDiffs[s][y,x], extremaValue))
            else:
                featuresInOctaveWithExtremas.append(f + (allDiffImages[octave][s][y, x], allDiffImages[octave][s][y,x]))
        featuresWithExtremas.append(np.array(featuresInOctaveWithExtremas))
    return featuresWithExtremas

def filterFeaturesAboveThreshold(features):
    featuresAboveThreshold = []
    for featuresInOctave in features:
        featuresAboveThreshold.append(featuresInOctave[np.abs(featuresInOctave[:,5] > THRESHOLD)])
    return featuresAboveThreshold

def filterFeatures(allDiffImages, features):
    print("== Filtering features...")

    featuresWithExtremas = computeExtremaValues(allDiffImages, features)

    featuresAboveThreshold = filterFeaturesAboveThreshold(featuresWithExtremas)
    print("Threshold filtered to %d" % len(featuresAboveThreshold))

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

    return (np.array(notEdges), np.array(edgesRemoved))

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
    (features2, edgesRemoved) = filterFeatures(allDiffImages, features)
    t2 = time.time()
    print("Filtered to %d features in %f (%d low-contrast removed, %d edges removed)" % (featuresAmount(features2), t2 - t1, len([]), featuresAmount(edgesRemoved)))

    drawFeatures(filename, features2, edgesRemoved)

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
