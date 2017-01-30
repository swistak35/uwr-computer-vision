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
from operator import itemgetter

WINDOW_RADIUS = 20 # So the window will be 41x41
ORIENTATION_BINS = 36
OCTAVES = 6
SCALES_PER_OCTAVE = 3
SIGMA = 1.6

ORIENTATION_BIN_ANGLE = 360.0 / ORIENTATION_BINS
SIGMA_K = np.power(2, 1.0 / SCALES_PER_OCTAVE)

COLORS = {
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
}

def saveFeatureDescriptors(imageFilename, featureDescriptors):
    basePath, extPath = os.path.splitext(imageFilename)
    scipy.io.savemat(basePath + "-features", {
            'features': featureDescriptors
        })

def findAllGaussianImages(image):
    # Would be nice to save all these images with respective features
    allGaussianImages = []
    for octave in range(OCTAVES):
        gaussianImages = []
        for scale in range(SCALES_PER_OCTAVE + 3):
            zoomedImage = sc.ndimage.zoom(image, np.power(0.5, octave))
            gaussianImage = sc.ndimage.filters.gaussian_filter(zoomedImage, SIGMA * np.power(SIGMA_K, scale), order = 0)
            gaussianImages.append(gaussianImage)
        allGaussianImages.append(gaussianImages)
    return allGaussianImages

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = COLORS['GREEN'], width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawFeature(draw, x, y, scale, octave, peaks, outlineColor = COLORS['GREEN']):
    realX = x * np.power(2, octave)
    realY = y * np.power(2, octave)
    radius = 3 + SIGMA * np.power(SIGMA_K, scale) * np.power(2, octave)
    drawSmallCircle(draw, realX, realY, width = radius, outlineColor = outlineColor)
    maxPeakV = peaks[0][1]
    if maxPeakV != 0: # how is that possible that there are zeros?
        normalizedPeaks = [ (i, v / maxPeakV) for (i,v) in peaks ]
        for (peakIndex, peakValue) in normalizedPeaks:
            lineLength = radius * peakValue
            angle = peakIndex * ORIENTATION_BIN_ANGLE
            draw.line([
                (realX, realY),
                (realX + lineLength * np.cos(np.radians(angle)), realY + lineLength * np.sin(np.radians(angle)))
            ])

def drawFeatures(filename, features):
    im = Image.open(filename)
    draw = ImageDraw.Draw(im)
    for f in features:
        (y, x, octave, scale, v, ev, peaks) = f
        drawFeature(draw, x, y, scale, octave, peaks)
    im.save(mkPath(filename, "-with-features"), "JPEG")

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

def siftDescriptor(fileset):
    (imageFilename, featuresFilename) = fileset

    features = sc.io.loadmat(featuresFilename)['fts']

    image = scipy.ndimage.imread(imageFilename, flatten = True)

    print("Finding gaussian images...")
    gaussianImages = findAllGaussianImages(image)

    print("Looking through features...")
    featureDescriptors = []
    featuresWithOrientation = []
    # for fi,f in enumerate(features[::10]):
    for fi,f in enumerate(features):
        (y, x, octave, scale, v, ev) = f
        octave = int(octave)
        scale = int(scale)
        y = int(y)
        x = int(x)
        image = gaussianImages[octave][scale]
        (height, width) = image.shape
        # TODO: Do something with that - add np.pad?
        if (x <= WINDOW_RADIUS + 1) or (x >= width - WINDOW_RADIUS - 1) or (y <= WINDOW_RADIUS + 1) or (y >= height - WINDOW_RADIUS - 1):
            continue

        window = image[(y - WINDOW_RADIUS - 1):(y + WINDOW_RADIUS + 2), (x - WINDOW_RADIUS - 1):(x + WINDOW_RADIUS + 2)]
        angles = np.empty((2 * WINDOW_RADIUS + 1, 2 * WINDOW_RADIUS + 1))
        magnitudes = np.empty((2 * WINDOW_RADIUS + 1, 2 * WINDOW_RADIUS + 1))
        for py in range(1, 2 * WINDOW_RADIUS + 2):
            for px in range(1, 2 * WINDOW_RADIUS + 2):
                # This double-loop can be optimized with element-wise numpy operations
                gx = window[py, px + 1] - window[py, px - 1]
                magnitudes[py - 1, px - 1] = np.sqrt(np.square(gx) + np.square(window[py+1, px] - window[py-1, px]))
                # https://en.wikipedia.org/wiki/Scale-invariant_feature_transform#Orientation_assignment
                # Tu nie powinno byc aby czasem angles[py, px] = ... ?
                angles[py - 1, px - 1] = 180.0 + np.degrees(np.arctan2((window[py+1, px] - window[py-1, px]), (window[py, px+1] - window[py, px-1])))
        # Times 1.5!
        magnitudesWithGaussian = sc.ndimage.filters.gaussian_filter(magnitudes, 1.5 * SIGMA * np.power(SIGMA_K, scale))

        (hist, bin_edges) = np.histogram(angles, bins = ORIENTATION_BINS, range = (0., 360.), weights = magnitudesWithGaussian)

        peaks = []
        for (i, v) in reversed(sorted(enumerate(hist), key = itemgetter(1) )):
            if peaks == []:
                peaks.append((i, v))
            elif len(peaks) < 4:
                if v > 0.8 * peaks[0][1]:
                    peaks.append((i, v))
        featuresWithOrientation.append((y, x, octave, scale, v, ev, peaks))
        # We could print here real coordinates, not on the gaussian image ones
        print("Point (%d, %d) -> Angles %s" % (x, y, peaks))

        for (currentAngleIndex, currentAngleMagnitude) in peaks:
            currentAngle = np.radians(currentAngleIndex * ORIENTATION_BIN_ANGLE)
            # Should be 18x18
            mw = np.arange(18) - 8.5
            (cx, cy) = np.meshgrid(mw, mw)
            pointsRaw = np.asfarray(np.stack((cx,cy), axis = 2).reshape(-1, 2))
            # pointsTranslated = pointsRaw + [x, y]
            # featurePatch1 = sc.ndimage.interpolation.map_coordinates(image, np.fliplr(pointsTranslated).T).reshape(18, 18)
            # sc.misc.imsave(mkPath(imageFilename, "-feature-%d-1" % fi), featurePatch1)

            agl = 2*np.pi - (np.pi / 2 + currentAngle)
            R = np.array([[np.cos(agl), -np.sin(agl)], [np.sin(agl), np.cos(agl)]])
            # pointsRotatedAndTranslated = pointsRaw.dot(R) + [x, y]
            # featurePatch2 = sc.ndimage.interpolation.map_coordinates(image, np.fliplr(pointsRotatedAndTranslated).T).reshape(18, 18)
            # sc.misc.imsave(mkPath(imageFilename, "-feature-%d-2" % fi), featurePatch2)

            pointsScaled = pointsRaw * np.power(SIGMA_K, scale) # scale? maybe scale-1 or scale+1?
            pointsScaledRotatedAndTranslated = pointsScaled.dot(R) + [x, y]
            featurePatch3 = sc.ndimage.interpolation.map_coordinates(image, np.fliplr(pointsScaledRotatedAndTranslated).T).reshape(18, 18)
            # sc.misc.imsave(mkPath(imageFilename, "-feature-%d-3" % fi), featurePatch3)

            finalPatch = featurePatch3

            # gradients = np.gradient(pointsScaledRotatedAndTranslated, axis = 0)
            histograms = np.empty((4,4,8))
            for ry in range(4):
                for rx in range(4):
                    subpatch = np.empty((4,4))
                    subpatchMagnitudes = np.empty((4,4))
                    for syt in range(4):
                        for sxt in range(4):
                            sy = 1 + ry*4 + syt
                            sx = 1 + rx*4 + sxt
                            subpatchMagnitudes[syt,sxt] = np.sqrt(np.square(finalPatch[sy, sx + 1] - finalPatch[sy, sx - 1]) + np.square(finalPatch[sy+1, sx] - finalPatch[sy-1, sx]))
                            subpatch[syt,sxt] = 180.0 + np.degrees(np.arctan2((finalPatch[sy+1, sx] - finalPatch[sy-1, sx]), (finalPatch[sy, sx+1] - finalPatch[sy, sx-1])))
                    subpatchMagnitudesWithGaussian = sc.ndimage.filters.gaussian_filter(subpatchMagnitudes, 8.0)
                    (hist, bin_edges) = np.histogram(subpatch, bins = 8, range = (0., 360.), weights = subpatchMagnitudes)
                    histograms[ry,rx] = hist
            descriptor = histograms.flatten()
            descriptorNormalized = descriptor / np.linalg.norm(descriptor)
            descriptorNormalized[descriptorNormalized > 0.2] = 0.2
            descriptorNormalized = descriptor / np.linalg.norm(descriptor)
            featureDescriptors.append(np.hstack(((y, x, octave, scale, currentAngle), descriptorNormalized)))
            # Gaussing after making a gradient or before?
            # What about this tri-linear interpolation thing

    # make the circle the same as there's scaling in feature descripting?
    # replace it with rectangles?
    drawFeatures(imageFilename, featuresWithOrientation)
    saveFeatureDescriptors(imageFilename, featureDescriptors)

def run():
    filesets = [
            # ("data/Notre Dame/1_o.jpg", "data/Notre Dame/1_o-featuresmat.mat"),
            # ("data/Notre Dame/2_o.jpg", "data/Notre Dame/2_o-featuresmat.mat"),
            ("data/duda/img_20170130_162706.jpg", "data/duda/img_20170130_162706-featuresmat.mat"),
            ("data/duda/c3bxl_zweaywcbm.jpg", "data/duda/c3bxl_zweaywcbm-featuresmat.mat"),
            ("data/fountain/0000.png", "data/fountain/0000-featuresmat.mat"),
            ("data/fountain/0001.png", "data/fountain/0001-featuresmat.mat"),
            # ("data/Mount Rushmore/9021235130_7c2acd9554_o.jpg", "data/Mount Rushmore/9021235130_7c2acd9554_o-featuresmat.mat"),
            # ("data/Mount Rushmore/9318872612_a255c874fb_o.jpg","data/Mount Rushmore/9318872612_a255c874fb_o-featuresmat.mat"),
            # ("data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg", "data/Episcopal Gaudi/3743214471_1b5bbfda98_o-featuresmat.mat"),
            # ("data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg", "data/Episcopal Gaudi/4386465943_8cf9776378_o-featuresmat.mat"),
        ]

    for fileset in filesets:
        print("=== Files: (%s, %s)" % fileset)
        siftDescriptor(fileset)

run()
