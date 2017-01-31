import scipy as sc
import scipy.io
import numpy as np
from colors import SiftSettings, GaussianImagesSet
from operator import itemgetter

class OrientationAssignment:
    def __init__(self, settings = SiftSettings()):
        self.settings = settings

    def compute(self, sourceImage, featuresFilename):
        WINDOW_RADIUS = self.settings.orientationWindowRadius
        ORIENTATION_BINS = 36
        ORIENTATION_BIN_ANGLE = 360.0 / ORIENTATION_BINS
        features = sc.io.loadmat(featuresFilename)['fts']

        # image = scipy.ndimage.imread(imageFilename, flatten = True)

        print("Finding gaussian images...")
        gaussianImages = GaussianImagesSet().compute(sourceImage)
        # gaussianImages = findAllGaussianImages(image)

        print("Looking through features...")
        featuresWithOrientation = []
        featuresWithOrientationToDraw = []
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
            magnitudesWithGaussian = sc.ndimage.filters.gaussian_filter(magnitudes, 1.5 * self.settings.sigma * np.power(self.settings.sigmaK, scale))

            (hist, bin_edges) = np.histogram(angles, bins = ORIENTATION_BINS, range = (0., 360.), weights = magnitudesWithGaussian)

            peaks = []
            for (i, v) in reversed(sorted(enumerate(hist), key = itemgetter(1) )):
                if peaks == []:
                    peaks.append((i, v))
                elif len(peaks) < 4:
                    if v > 0.8 * peaks[0][1]:
                        peaks.append((i, v))
            # We could print here real coordinates, not on the gaussian image ones
            # print("Point (%d, %d) -> Angles %s" % (x, y, peaks))

            featuresWithOrientationToDraw.append((y, x, octave, scale, v, ev, peaks))
            for (currentAngleIndex, currentAngleMagnitude) in peaks:
                currentAngle = np.radians(currentAngleIndex * ORIENTATION_BIN_ANGLE)
                featuresWithOrientation.append((y, x, octave, scale, v, ev, currentAngle))

        # drawFeatures(imageFilename, featuresWithOrientationToDraw)
        return (featuresWithOrientation, featuresWithOrientationToDraw)
