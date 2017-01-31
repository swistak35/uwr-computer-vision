import numpy as np
import scipy as sc
import scipy.ndimage
from colors import SiftSettings, GaussianImagesSet

# TODO: Make it able to share GaussianImageSet
# TODO: In `pointsScaled` self.settings.sigma should be too?
# TODO: Should illumination be here, or before computing histogram?
# TODO: Tri-linear interpolation
# TODO: Add drawing with rectangles

class FeatureDescripting:
    def __init__(self, settings = SiftSettings()):
        self.settings = settings

    def saveFeatureDescriptors(self, outputFilename, featureDescriptors):
        scipy.io.savemat(outputFilename, {
            'features': featureDescriptors
        })

    def compute(self, sourceImage, features):
        gaussianImages = GaussianImagesSet().compute(sourceImage)

        featureDescriptors = []
        for f in features:
            (y, x, octave, scale, v, ev, currentAngle) = f
            octave, scale = int(octave), int(scale)
            y, x = int(y), int(x)
            image = gaussianImages[octave][scale]
            mw = np.arange(18) - 8.5
            (cx, cy) = np.meshgrid(mw, mw)
            pointsRaw = np.asfarray(np.stack((cx,cy), axis = 2).reshape(-1, 2))

            agl = 2*np.pi - (np.pi / 2 + currentAngle)
            R = np.array([[np.cos(agl), -np.sin(agl)], [np.sin(agl), np.cos(agl)]])

            pointsScaled = pointsRaw * np.power(self.settings.sigmaK, scale)
            pointsScaledRotatedAndTranslated = pointsScaled.dot(R) + [x, y]
            finalPatch = sc.ndimage.interpolation.map_coordinates(image, np.fliplr(pointsScaledRotatedAndTranslated).T).reshape(18, 18)
            # sc.misc.imsave(mkPath(imageFilename, "-feature-%d-3" % fi), featurePatch3)

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

            # Normalization
            descriptorNormalized = descriptor / np.linalg.norm(descriptor)

            # Illumination protection
            descriptorNormalized[descriptorNormalized > 0.2] = 0.2
            descriptorNormalized = descriptor / np.linalg.norm(descriptor)

            featureDescriptors.append(np.hstack(((y, x, octave, scale, currentAngle), descriptorNormalized)))

        return np.array(featureDescriptors)
