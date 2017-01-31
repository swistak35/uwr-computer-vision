import scipy as sc
import scipy.io
import numpy as np
import PIL as pil
import Image, ImageDraw
from colors import getRandomColor, COLORS, debug, drawCircle, drawLineUnderAngle, SiftSettings

# TODO: There should be an option to display only "50 best" matches
# TODO: It (removing because of ratio) could be done different way (better?) with cosine, like in paper
# TODO: Detector moglby przesuwac ten punkt nie po prostu * octaves, ale jeszcze wziac pod uwage ze ten pixel reprezentuje jakies 8, wiec walnac go w srodek

class FeatureMatching:
    DEFAULT_RATIO_THRESHOLD = 0.75 # Setting to 1.0 disables check
    DEFAULT_SANITY_THRESHOLD = 2.0

    def __init__(self, settings = SiftSettings()):
        self.matches = None
        self.settings = settings

    # Drawing
    def featureForDrawing(self, f, offset):
        (y, x, octave, scale, angle) = f
        realX = x * np.power(2, octave) + offset
        realY = y * np.power(2, octave)
        radius = 3 + self.settings.sigma * np.power(self.settings.sigmaK, scale) * np.power(2, octave)
        angleDeg = np.degrees(angle)
        return (realX, realY, radius, angleDeg)

    def drawMatch(self, draw, m, offset):
        (rx1, ry1, radius1, angle1) = self.featureForDrawing(m[0][0:5], 0)
        (rx2, ry2, radius2, angle2) = self.featureForDrawing(m[1][0:5], offset)
        color = getRandomColor()
        drawCircle(draw, rx1, ry1, width = radius1, outlineColor = color)
        drawLineUnderAngle(draw, rx1, ry1, radius1, angle1)
        drawCircle(draw, rx2, ry2, width = radius2, outlineColor = color)
        drawLineUnderAngle(draw, rx2, ry2, radius2, angle2)
        draw.line([(rx1, ry1), (rx2, ry2)], fill = color, width = 1)

    def drawMatchesGeneric(self, outputFilename, matches):
        im1 = Image.open(self.imageFilename1)
        im2 = Image.open(self.imageFilename2)
        totalWidth = im1.size[0] + im2.size[0]
        maxHeight = max(im1.size[1], im2.size[1])
        imOut = Image.new('RGB', (totalWidth, maxHeight))
        offset = im1.size[0]
        imOut.paste(im1, (0, 0))
        imOut.paste(im2, (offset, 0))
        draw = ImageDraw.Draw(imOut)
        for m in matches:
            self.drawMatch(draw, m, offset)
        imOut.save(outputFilename, "JPEG")

    def drawMatches(self, outputFilename):
        self.drawMatchesGeneric(outputFilename, self.matches)

    def drawTopMatches(self, outputFilename, amount = 20):
        topMatches = self.getTopMatches(amount = amount)
        self.drawMatchesGeneric(outputFilename, topMatches)

    # Finding matches
    def compute(self, imageFilename1, imageFilename2, featuresFilename1, featuresFilename2,
            ratioThreshold = DEFAULT_RATIO_THRESHOLD,
            sanityThreshold = DEFAULT_SANITY_THRESHOLD):
        self.imageFilename1 = imageFilename1
        self.imageFilename2 = imageFilename2

        features1 = sc.io.loadmat(featuresFilename1)['features']
        features2 = sc.io.loadmat(featuresFilename2)['features']

        matches = []
        for fi, f in enumerate(features1):
            descriptor1 = f[5:]
            descriptors2 = features2[:,5:]
            distances = np.abs(descriptors2 - descriptor1).sum(axis = 1)
            (bestIdx, secondIdx) = distances.argsort()[:2]
            bestValue = distances[bestIdx]
            if bestValue > sanityThreshold:
                continue
            if (bestValue / distances[secondIdx]) > ratioThreshold:
                continue
            matches.append((f, features2[bestIdx], bestValue))
            debug("Found match for %d, value: %f" % (fi, bestValue))

        print("Found %d matches" % (len(matches)))
        self.matches = np.array(matches)

    def getTopMatches(self, amount = 20):
        return self.matches[self.matches[:,2].argsort()][0:amount]

    # Verification
    def verify(self, verificationFile):
        mats = sc.io.loadmat(verificationFile)
        verifiedFeatures = np.hstack((mats['x1'], mats['y1'], mats['x2'], mats['y2']))

        verifiedCounter = 0
        verifiedFailedCounter = 0
        notFoundCounter = 0
        otherCounter = 0
        totalCounter = len(self.matches)

        for (f1, f2, v) in self.matches:
            (y1, x1) = f1[0:2]
            (y2, x2) = f2[0:2]

            norms = np.linalg.norm(verifiedFeatures[:, 0:2] - (x1, y1), axis = 1)
            foundNorms = verifiedFeatures[norms < 5.0]
            if foundNorms.shape[0] == 0:
                notFoundCounter += 1
                debug("Verification: didn't find this feature in verification table")
            else:
                results = np.linalg.norm(foundNorms[:, 2:4] - (x2, y2), axis=1)
                verifiedResults = foundNorms[results < 5.0]
                if foundNorms.shape[0] == 1:
                    if verifiedResults.shape[0] == 1:
                        verifiedCounter += 1
                        debug("Verification: Found and verified exactly one feature")
                    else:
                        verifiedFailedCounter += 1
                        debug("Verification: Found exactly one feature, but didn't verified it (distance: %f)" % (results[0]))
                else:
                    otherCounter += 1
                    debug("Verification: Found multiple features")

        print("=== Verification complete.")
        print("Verified:  %d" % verifiedCounter)
        print("Failed:    %d" % verifiedFailedCounter)
        print("Not found: %d" % notFoundCounter)
        print("Other:     %d" % otherCounter)
        print("Total:     %d" % totalCounter)











def run():
    filesets = [
            ("data/Notre Dame/1_o.jpg", "data/Notre Dame/2_o.jpg",
                "data/Notre Dame/1_o-features.mat", "data/Notre Dame/2_o-features.mat",
                "data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat"),
            # ("data/duda/img_20170130_162706.jpg", "data/duda/c3bxl_zweaywcbm.jpg", None),
            # ("data/fountain/0000.png", "data/fountain/0001.png", None),
        ]

    for fileset in filesets:
        print("=== Files: (%s, %s)" % fileset[0:2])
        fm = FeatureMatching()
        fm.compute(fileset[0], fileset[1], fileset[2], fileset[3])
        fm.drawTopMatches("matches.jpg", amount = 60)
        fm.verify(fileset[4])

run()
