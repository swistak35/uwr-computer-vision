import scipy as sc
import scipy.io
import scipy.misc
import scipy.ndimage
import numpy as np
# import numpy.linalg
import PIL as pil
import os.path
import Image, ImageDraw
import time

octaves = 4 # Should be 4 in final
scalesPerOctave = 3
sigma = 1.6
k = np.power(2, 1.0 / scalesPerOctave)

# How to draw the circle based on the sigma?

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = (0, 255, 0), width = 2):
#     # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawFeatures(filename, features):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for (y, x, octave, scale) in features:
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

def siftCornerDetector(filename):
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

    t1 = time.time()
    features = findFeatures2(allDiffImages)
    t2 = time.time()
    print("Found %d features" % len(features))
    print("Finished finding features in %f" % (t2 - t1))

    drawFeatures(filename, features)


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
        siftCornerDetector(filename)

run()
