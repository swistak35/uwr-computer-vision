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

WINDOW_RADIUS = 20 # So the window will be 41x41
ORIENTATION_BINS = 36
OCTAVES = 4
SCALES_PER_OCTAVE = 3
SIGMA = 1.6

ORIENTATION_BIN_ANGLE = 360.0 / ORIENTATION_BINS
SIGMA_K = np.power(2, 1.0 / SCALES_PER_OCTAVE)

COLORS = {
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
}

def findAllGaussianImages(image):
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
    normalizedPeaks = [ (i, v / maxPeakV) for (i,v) in peaks ]
    for (peakIndex, peakValue) in normalizedPeaks:
        lineLength = radius * peakValue
        angle = peakIndex * 10.0
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

    # featuresMatFilename1 = "data/Notre Dame/1_o-featuresmat.mat"
    # featuresMatFilename2 = "data/Notre Dame/2_o-featuresmat.mat"

    features1 = sc.io.loadmat(featuresFilename)['fts']
    # features2 = sc.io.loadmat(featuresMatFilename2)['fts']

    # filename1 = "data/Notre Dame/1_o.jpg"
    # filename2 = "data/Notre Dame/2_o.jpg"

    image1 = scipy.ndimage.imread(imageFilename, flatten = True)
    gaussianImages1 = findAllGaussianImages(image1)

    print("Looking through features...")
    featuresWithOrientation = []
    for f in features1:
        (y, x, octave, scale, v, ev) = f
        octave = int(octave)
        scale = int(scale)
        y = int(y)
        x = int(x)
        image = gaussianImages1[octave][scale]
        (height, width) = image.shape
        if (x <= WINDOW_RADIUS + 1) or (x >= width - WINDOW_RADIUS - 1) or (y <= WINDOW_RADIUS + 1) or (y >= height - WINDOW_RADIUS - 1):
            continue

        obins = [ [] for i in range(ORIENTATION_BINS) ]
        window = image[(y - WINDOW_RADIUS - 1):(y + WINDOW_RADIUS + 2), (x - WINDOW_RADIUS - 1):(x + WINDOW_RADIUS + 2)]
        # convolve window with the gaussian here
        for py in range(1, 2 * WINDOW_RADIUS + 1):
            for px in range(1, 2 * WINDOW_RADIUS + 1):
                magnitude = np.sqrt(np.square(window[py, px+1] - window[py, px-1]) + np.square(window[py+1, px] - window[py-1, px]))
                # https://en.wikipedia.org/wiki/Scale-invariant_feature_transform#Orientation_assignment
                angle = 180.0 + np.degrees(np.arctan2((window[py+1, px] - window[py-1, px]), (window[py, px+1] - window[py, px-1])))
                # print("Point (%d, %d) -> (%d, %d) | Magnitude: %f | Angle: %f" % (x, y, px, py, magnitude, angle))
                bin_number = int(angle / (360.0 / ORIENTATION_BINS))
                if bin_number == ORIENTATION_BINS: # Better: if angle == 360.0
                    bin_number = ORIENTATION_BINS - 1
                assert(bin_number >= 0 and bin_number < ORIENTATION_BINS)
                obins[bin_number].append((px, py, angle, magnitude))
        sortedObins = sorted([ (i, data) for (i, data) in enumerate(obins) ], key = lambda (i,vs): sum([ ii[3] for ii in vs ]) )
        peaks = []
        for (i, es) in reversed(sortedObins):
            currentSum = sum([ ii[3] for ii in es ])
            if peaks == []:
                peaks.append((i, currentSum))
            elif len(peaks) < 4:
                lastPeak = peaks[-1]
                if currentSum > 0.8 * lastPeak[1]:
                    peaks.append((i, currentSum))
        # peaks = [(max_s, max_i)]
        featuresWithOrientation.append((y, x, octave, scale, v, ev, peaks))
        print("Point (%d, %d) -> Angles %s" % (x, y, peaks))

    drawFeatures(imageFilename, featuresWithOrientation)

def run():
    filesets = [
            ("data/Notre Dame/1_o.jpg", "data/Notre Dame/1_o-featuresmat.mat"),
            ("data/Notre Dame/2_o.jpg", "data/Notre Dame/2_o-featuresmat.mat"),
            # "data/Mount Rushmore/9021235130_7c2acd9554_o.jpg",
            # "data/Mount Rushmore/9318872612_a255c874fb_o.jpg",
            # "data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg",
            # "data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg",
        ]

    for fileset in filesets:
        print("=== Files: (%s, %s)" % fileset)
        siftDescriptor(fileset)

run()
