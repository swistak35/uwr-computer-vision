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

# How to draw the circle based on the sigma?
# TODO: Draw the filtered features

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = (0, 255, 0), width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawFeatures(filename, features):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for (y, x, octave, scale, v, ev) in features:
        realX = x * np.power(2, octave)
        realY = y * np.power(2, octave)
        # ...
        radius = 5 + sigma * np.power(k, octave * scalesPerOctave + scale)
        drawSmallCircle(draw, realX, realY, width = radius)
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

def findFeatures2(allDiffImages):
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

    features = []
    for octave in range(octaves):
        (height, width) = allDiffImages[octave][0].shape
        for scale in range(1, scalesPerOctave + 1):
            print("Searching octave %d scale %d" % (octave, scale))
            for y in range(1, height - 2):
                for x in range(1, width - 2):
                    belowImageV = allDiffImages[octave][scale - 1][y,x]
                    currentImageV = allDiffImages[octave][scale][y,x]
                    aboveImageV = allDiffImages[octave][scale + 1][y,x]
                    if minImages[octave][scale - 1][y-1,x-1] > currentImageV and minImages[octave][scale + 1][y-1,x-1] > currentImageV and minImages[octave][scale][y-1,x-1] > currentImageV and belowImageV > currentImageV and aboveImageV > currentImageV:
                        features.append((y, x, octave, scale))
                    if maxImages[octave][scale - 1][y-1,x-1] < currentImageV and maxImages[octave][scale + 1][y-1,x-1] < currentImageV and maxImages[octave][scale][y-1,x-1] < currentImageV and belowImageV < currentImageV and aboveImageV < currentImageV:
                        features.append((y, x, octave, scale))
    return features

def dxy(images, x, y, s):
    d1 = (images[s][y+1, x+1] - images[s][y+1,x-1]) / 2
    d2 = (images[s][y-1, x+1] - images[s][y-1,x-1]) / 2
    return (d1 - d2) / 2

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

def filterFeatures(allDiffImages, features):
    filteredFeatures = []
    flattenedDiffs = np.array(allDiffImages[0])
    fullHessian = hessian(flattenedDiffs)
    for f in features:
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
            filteredFeatures.append(f + (flattenedDiffs[s][y,x], extremaValue))
        else:
            filteredFeatures.append(f + (allDiffImages[octave][s][y, x], allDiffImages[octave][s][y,x]))
    filteredFeatures = np.array(filteredFeatures)
    featuresAboveThreshold = filteredFeatures[np.abs(filteredFeatures[:,5] > THRESHOLD)]

    print("Threshold filtered to %d" % len(featuresAboveThreshold))

    hessian2x2 = []
    for img in flattenedDiffs:
        hessian2x2.append(hessian(img))
    moreFiltered = []
    for f in featuresAboveThreshold:
        (y, x, octave, s, v, ev) = f
        if octave == 0:
            hs = hessian2x2[int(s)][:,:,y,x] # How int rounds? Potential fuckup?
            r = np.power(np.trace(hs), 2) / np.linalg.det(hs)
            edge_ratio_coefficient = np.power(EDGE_RATIO + 1, 2) / EDGE_RATIO
            if r < edge_ratio_coefficient:
                moreFiltered.append(f)
        else:
            moreFiltered.append(f)

    return np.array(moreFiltered)


def findDiffImages(filename):
    image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale

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
            sc.misc.imsave(mkPath(filename, "-1-sift-scale-%d-%d" % (octave, scale)), gaussianImage)
            if scale > 0:
                diffGaussian = gaussianImage - gaussianImages[-1]
                sc.misc.imsave(mkPath(filename, "-2-sift-diff-%d-%d" % (octave, scale)), diffGaussian)
                diffImages.append(diffGaussian)
            gaussianImages.append(gaussianImage)
        allImages.append(gaussianImages)
        allDiffImages.append(diffImages)
    return allDiffImages

def siftCornerDetector(filename):
    allDiffImages = findDiffImages(filename)

    flattenedDiffs = []
    for s in range(scalesPerOctave+2):
        flattenedDiffs.append(allDiffImages[0][s])
    flattenedDiffs = np.array(flattenedDiffs)

    t1 = time.time()
    features = findFeatures2(allDiffImages)
    t2 = time.time()
    print("Found %d features" % len(features))
    print("Finished finding features in %f" % (t2 - t1))

    print("Filtering features...")
    t1 = time.time()
    features2 = filterFeatures(allDiffImages, features)
    t2 = time.time()
    print("Filtered to %d features" % len(features2))
    print("Finished finding features in %f" % (t2 - t1))

    drawFeatures(filename, features2)


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
