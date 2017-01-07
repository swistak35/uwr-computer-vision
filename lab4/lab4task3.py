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

octaves = 2 # Should be 4 in final
scalesPerOctave = 3
sigma = 1.6
k = 1.41

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
    previousfeatures = 0
    features = []
    for octave in range(octaves):
        for scale in range(1, scalesPerOctave + 1):
            print("Searching octave %d scale %d" % (octave, scale))
            t1 = time.time()
            belowImage = allDiffImages[octave][scale - 1]
            currentImage = allDiffImages[octave][scale]
            aboveImage = allDiffImages[octave][scale + 1]
            (height, width) = belowImage.shape
            belowMins = np.amin(np.array([
                belowImage[1:(height - 1),1:(width - 1)],

                belowImage[0:(height - 2),1:(width - 1)],
                belowImage[2:(height),1:(width - 1)],
                belowImage[1:(height - 1),0:(width - 2)],
                belowImage[1:(height - 1),2:(width)],

                belowImage[0:(height - 2),0:(width - 2)],
                belowImage[0:(height - 2),2:(width)],
                belowImage[2:(height),0:(width - 2)],
                belowImage[2:(height),2:(width)],
                ]), axis = 0)
            for y in range(1, currentImage.shape[0] - 2):
                for x in range(1, currentImage.shape[1] - 2):
                    # belowWindow = belowImage[y-1:y+2,x-1:x+2]
                    # assert(belowWindow.shape == (3,3))
                    aboveWindow = aboveImage[y-1:y+2,x-1:x+2]
                    assert(aboveWindow.shape == (3,3))
                    currentWindow = mkCurrentWindow(currentImage, y, x)
                    if belowMins[y-1,x-1] > currentImage[y,x] and np.all(aboveWindow > currentImage[y,x]) and np.all(currentWindow > currentImage[y,x]):
                        features.append((y, x, octave, scale))
                    # if np.all(belowWindow > currentImage[y,x]) and np.all(aboveWindow > currentImage[y,x]) and np.all(currentWindow > currentImage[y,x]):
                        # features.append((y, x, octave, scale))
                    # if np.all(belowWindow < currentImage[y,x]) and np.all(aboveWindow < currentImage[y,x]) and np.all(currentWindow < currentImage[y,x]):
                        # features.append((y, x, octave, scale))
            t2 = time.time()
            print("Finished in %f" % (t2 - t1))
            print("Found %d new features" % (len(features) - previousfeatures))
            previousfeatures = len(features)
    print("Found %d features" % len(features))

def siftCornerDetector(filename):
    image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale

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

    findFeatures1(allDiffImages)


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
