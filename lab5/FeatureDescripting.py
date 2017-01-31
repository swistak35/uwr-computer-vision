import numpy as np
import scipy as sc
import scipy.ndimage
from colors import SiftSettings

import os.path

def findAllGaussianImages(image):
    settings = SiftSettings()
    OCTAVES = settings.octaves
    SCALES_PER_OCTAVE = settings.scalesPerOctave
    SIGMA = settings.sigma
    SIGMA_K = settings.sigmaK
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

class FeatureDescripting:
    def __init__(self):
        self.descriptors = None
        self.settings = SiftSettings()

    def saveFeatureDescriptors(self, imageFilename, featureDescriptors):
        basePath, extPath = os.path.splitext(imageFilename)
        scipy.io.savemat(basePath + "-features", {
                'features': featureDescriptors
            })

    def compute(self, sourceImageFilename, features):
        sourceImage = sc.ndimage.imread(sourceImageFilename, flatten = True)
        gaussianImages = findAllGaussianImages(sourceImage)
        featureDescriptors = []
        for f in features:
            (y, x, octave, scale, v, ev, currentAngle) = f
            octave = int(octave)
            scale = int(scale)
            y = int(y)
            x = int(x)
            image = gaussianImages[octave][scale]
            # Should be 18x18
            mw = np.arange(18) - 8.5
            (cx, cy) = np.meshgrid(mw, mw)
            pointsRaw = np.asfarray(np.stack((cx,cy), axis = 2).reshape(-1, 2))

            agl = 2*np.pi - (np.pi / 2 + currentAngle)
            R = np.array([[np.cos(agl), -np.sin(agl)], [np.sin(agl), np.cos(agl)]])

            pointsScaled = pointsRaw * np.power(self.settings.sigmaK, scale) # scale? maybe scale-1 or scale+1?
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
        self.saveFeatureDescriptors(sourceImageFilename, featureDescriptors)
