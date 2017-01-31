import random
import numpy as np

COLORS = {
    'RED':    (255, 0,   0),
    'GREEN':  (0,   255, 0),
    'BLUE':   (0,   0,   255),
    'DUNNO1': (0,   255, 255),
    'DUNNO2': (255, 0,   255),
    'DUNNO3': (255, 255, 0),
    'BLACK':  (0, 0, 0),
}

def getRandomColor():
    return random.choice(COLORS.values())

def drawCircle(draw, x, y, fillColor = None, outlineColor = COLORS['BLACK'], width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def drawLineUnderAngle(draw, x, y, length, angle):
    draw.line([
        (x, y),
        (x + length * np.cos(np.radians(angle)), y + length * np.sin(np.radians(angle)))
    ])

DEBUG = False

def debug(msg):
    if DEBUG:
        print("[DEBUG] %s" % msg)


class SiftSettings:
    def __init__(self):
        self.sigma = 1.6
        self.scalesPerOctave = 3
        self.octaves = 4
        self.orientationWindowRadius = 20
        self.sigmaK = np.power(2, 1.0 / self.scalesPerOctave)
        self.orientationBins = 36
        self.orientationBinAngle = 360.0 / self.orientationBins

import scipy as sc
import scipy.ndimage

class GaussianImagesSet:
    def __init__(self, settings = SiftSettings()):
        self.settings = settings

    def compute(self, sourceImage):
        allGaussianImages = []
        for octave in range(self.settings.octaves):
            gaussianImages = []
            for scale in range(self.settings.scalesPerOctave + 3):
                zoomedImage = sc.ndimage.zoom(sourceImage, np.power(0.5, octave))
                gaussianImage = sc.ndimage.filters.gaussian_filter(zoomedImage, self.settings.sigma * np.power(self.settings.sigmaK, scale), order = 0)
                gaussianImages.append(gaussianImage)
            allGaussianImages.append(gaussianImages)
        return allGaussianImages
