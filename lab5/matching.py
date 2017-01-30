import scipy as sc
import scipy.io
import numpy as np
import os.path
import PIL as pil
import Image, ImageDraw
import random

SIGMA = 1.6
OCTAVES = 4
SCALES_PER_OCTAVE = 3
SIGMA_K = np.power(2, 1.0 / SCALES_PER_OCTAVE)

SANITY_THRESHOLD = 2.0

COLORS = {
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'DUNNO1': (0, 255, 255),
    'DUNNO2': (255, 0, 255),
    'DUNNO3': (255, 255, 0),
}

# TODO: There should be an option to display only "50 best" matches

def getRandomColor():
    return random.choice(COLORS.values())

def drawSmallCircle(draw, x, y, fillColor = None, outlineColor = COLORS['GREEN'], width = 2):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

def featureForDrawing(draw, x, y, scale, octave, angle, offset):
    realX = x * np.power(2, octave) + offset
    realY = y * np.power(2, octave)
    radius = 3 + SIGMA * np.power(SIGMA_K, scale) * np.power(2, octave)
    angleDeg = np.degrees(angle)
    return (realX, realY, radius, angleDeg)
    # maxPeakV = peaks[0][1]
    # if maxPeakV != 0: # how is that possible that there are zeros?
    #     normalizedPeaks = [ (i, v / maxPeakV) for (i,v) in peaks ]
    #     for (peakIndex, peakValue) in normalizedPeaks:
    #         lineLength = radius * peakValue
    #         angle = peakIndex * ORIENTATION_BIN_ANGLE
    #         draw.line([
    #             (realX, realY),
    #             (realX + lineLength * np.cos(np.radians(angle)), realY + lineLength * np.sin(np.radians(angle)))
    #         ])

def drawMatch(draw, m, offset):
    (y1, x1, octave1, scale1, currentAngle1) = m[0][0:5]
    (y2, x2, octave2, scale2, currentAngle2) = m[1][0:5]
    (rx1, ry1, radius1, angle1) = featureForDrawing(draw, x1, y1, scale1, octave1, currentAngle1, 0)
    (rx2, ry2, radius2, angle2) = featureForDrawing(draw, x2, y2, scale2, octave2, currentAngle2, offset)
    color = getRandomColor()
    drawSmallCircle(draw, rx1, ry1, width = radius1, outlineColor = color)
    drawSmallCircle(draw, rx2, ry2, width = radius2, outlineColor = color)
    # TODO: Draw orientation
    draw.line([
            (rx1, ry1),
            (rx2, ry2),
        ], fill = color, width = 1)

def drawMatches(imageFilename1, imageFilename2, matches):
    im1 = Image.open(imageFilename1)
    im2 = Image.open(imageFilename2)
    totalWidth = im1.size[0] + im2.size[0]
    maxHeight = max(im1.size[1], im2.size[1])
    imOut = Image.new('RGB', (totalWidth, maxHeight))
    offset = im1.size[0]
    imOut.paste(im1, (0, 0))
    imOut.paste(im2, (offset, 0))
    draw = ImageDraw.Draw(imOut)
    for m in matches:
        drawMatch(draw, m, offset)
    # TODO: Don't write to the same file
    imOut.save("matches.jpg", "JPEG")

def matchImages(imageFilename1, imageFilename2, verificationFile):
    basePath1, extPath1 = os.path.splitext(imageFilename1)
    basePath2, extPath2 = os.path.splitext(imageFilename2)
    matFilename1 = basePath1 + "-features.mat"
    matFilename2 = basePath2 + "-features.mat"
    features1 = sc.io.loadmat(matFilename1)['features']
    features2 = sc.io.loadmat(matFilename2)['features']

    mats = sc.io.loadmat(verificationFile)
    verifiedFeatures = np.hstack((mats['x1'], mats['y1'], mats['x2'], mats['y2']))

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

        norms = np.linalg.norm(verifiedFeatures[:, 0:2] - (x, y), axis = 1)
        foundNorms = verifiedFeatures[norms < 5.0]
        if foundNorms.shape[0] == 0:
            print("Verification: didn't find this feature in verification table")
        else:
            f2 = (features2[bestIdx][1], features2[bestIdx][0])
            results = np.linalg.norm(foundNorms[:, 2:4] - f2, axis=1)
            verifiedResults = foundNorms[results < 5.0]
            if foundNorms.shape[0] == 1:
                if verifiedResults.shape[0] == 1:
                    print("Verification: Found and verified exactly one feature")
                else:
                    print("Verification: Found exactly one feature, but didn't verified it (distance: %f)" % (results[0]))
            else:
                print("Verification: Found multiple features")

    print("Found %d matches" % (len(matches)))
    drawMatches(imageFilename1, imageFilename2, matches)


def run():
    filesets = [
            ("data/Notre Dame/1_o.jpg", "data/Notre Dame/2_o.jpg", "data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat"),
            # ("data/duda/img_20170130_162706.jpg", "data/duda/c3bxl_zweaywcbm.jpg"),
            # ("data/Mount Rushmore/9021235130_7c2acd9554_o.jpg", "data/Mount Rushmore/9021235130_7c2acd9554_o-featuresmat.mat"),
            # ("data/Mount Rushmore/9318872612_a255c874fb_o.jpg","data/Mount Rushmore/9318872612_a255c874fb_o-featuresmat.mat"),
            # ("data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg", "data/Episcopal Gaudi/3743214471_1b5bbfda98_o-featuresmat.mat"),
            # ("data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg", "data/Episcopal Gaudi/4386465943_8cf9776378_o-featuresmat.mat"),
        ]

    for fileset in filesets:
        print("=== Files: (%s, %s)" % fileset[0:2])
        matchImages(fileset[0], fileset[1], fileset[2])

run()
