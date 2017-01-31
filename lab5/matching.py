import scipy as sc
import scipy.io
import numpy as np
import os.path
import PIL as pil
import Image, ImageDraw
from colors import getRandomColor, COLORS, debug, drawCircle, drawLineUnderAngle

SIGMA = 1.6
OCTAVES = 4
SCALES_PER_OCTAVE = 3
SIGMA_K = np.power(2, 1.0 / SCALES_PER_OCTAVE)

SANITY_THRESHOLD = 3.0

# TODO: There should be an option to display only "50 best" matches

class FeatureMatching:
    def __init__(self):
        self.matches = None

    # Drawing
    def featureForDrawing(self, f, offset):
        (y, x, octave, scale, angle) = f
        realX = x * np.power(2, octave) + offset
        realY = y * np.power(2, octave)
        radius = 3 + SIGMA * np.power(SIGMA_K, scale) * np.power(2, octave)
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

    def drawMatches(self, outputFilename):
        im1 = Image.open(self.imageFilename1)
        im2 = Image.open(self.imageFilename2)
        totalWidth = im1.size[0] + im2.size[0]
        maxHeight = max(im1.size[1], im2.size[1])
        imOut = Image.new('RGB', (totalWidth, maxHeight))
        offset = im1.size[0]
        imOut.paste(im1, (0, 0))
        imOut.paste(im2, (offset, 0))
        draw = ImageDraw.Draw(imOut)
        for m in self.matches:
            self.drawMatch(draw, m, offset)
        imOut.save(outputFilename, "JPEG")

    def compute(self, imageFilename1, imageFilename2):
        self.imageFilename1 = imageFilename1
        self.imageFilename2 = imageFilename2

        basePath1, extPath1 = os.path.splitext(imageFilename1)
        basePath2, extPath2 = os.path.splitext(imageFilename2)
        matFilename1 = basePath1 + "-features.mat"
        matFilename2 = basePath2 + "-features.mat"
        features1 = sc.io.loadmat(matFilename1)['features']
        features2 = sc.io.loadmat(matFilename2)['features']

        matches = []
        for fi, f in enumerate(features1):
            (y, x, octave, scale, angle) = f[0:5]
            descriptor = f[5:]
            descriptors2 = features2[:,5:]
            distances = np.abs(descriptors2 - descriptor).sum(axis = 1)
            (bestIdx, secondIdx) = distances.argsort()[:2]
            bestValue = distances[bestIdx]
            secondValue = distances[secondIdx]
            if bestValue > SANITY_THRESHOLD:
                continue
            # TODO: It could be done different way (better?) with cosine, like in paper
            if (bestValue / secondValue) > 0.75:
                continue
            print("Found match for %d, value: %f" % (fi,bestValue))
            matches.append((f, features2[bestIdx]))


        print("Found %d matches" % (len(matches)))
        self.matches = matches

    def verify(self, verificationFile):
        mats = sc.io.loadmat(verificationFile)
        verifiedFeatures = np.hstack((mats['x1'], mats['y1'], mats['x2'], mats['y2']))

        verifiedCounter = 0
        verifiedFailedCounter = 0
        notFoundCounter = 0
        otherCounter = 0
        totalCounter = len(self.matches)

        for (f1, f2) in self.matches:
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
            ("data/Notre Dame/1_o.jpg", "data/Notre Dame/2_o.jpg", "data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat"),
            # ("data/duda/img_20170130_162706.jpg", "data/duda/c3bxl_zweaywcbm.jpg", None),
            # ("data/fountain/0000.png", "data/fountain/0001.png", None),
        ]

    for fileset in filesets:
        print("=== Files: (%s, %s)" % fileset[0:2])
        fm = FeatureMatching()
        fm.compute(fileset[0], fileset[1])
        fm.drawMatches("matches.jpg")
        fm.verify(fileset[2])

run()
