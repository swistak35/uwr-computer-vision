import numpy as np
from colors import COLORS, SiftSettings
import PIL as pil
import Image, ImageDraw

class Drawing:
    def drawFeaturesWithOrientations(self, inputFilename, outputFilename, features):
        im = Image.open(inputFilename)
        draw = ImageDraw.Draw(im)
        for f in features:
            (y, x, octave, scale, v, ev, peaks) = f
            self.drawFeature(draw, x, y, scale, octave, peaks)
        im.save(outputFilename, "JPEG")

    def drawSmallCircle(self, draw, x, y, fillColor = None, outlineColor = COLORS['GREEN'], width = 2):
        # Top left and bottom right corners
        draw.ellipse((x - width, y - width, x + width, y + width), fill = fillColor, outline = outlineColor)

    def drawFeature(self, draw, x, y, scale, octave, peaks, outlineColor = COLORS['GREEN'], siftSettings = SiftSettings()):
        realX = x * np.power(2, octave)
        realY = y * np.power(2, octave)
        radius = 3 + siftSettings.sigma * np.power(siftSettings.sigmaK, scale) * np.power(2, octave)
        self.drawSmallCircle(draw, realX, realY, width = radius, outlineColor = outlineColor)
        maxPeakV = peaks[0][1]
        if maxPeakV != 0: # how is that possible that there are zeros?
            normalizedPeaks = [ (i, v / maxPeakV) for (i,v) in peaks ]
            for (peakIndex, peakValue) in normalizedPeaks:
                lineLength = radius * peakValue
                angle = peakIndex * siftSettings.orientationBinAngle
                draw.line([
                    (realX, realY),
                    (realX + lineLength * np.cos(np.radians(angle)), realY + lineLength * np.sin(np.radians(angle)))
                ])
