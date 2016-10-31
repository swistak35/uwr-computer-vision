import numpy as np
import os.path

# Build projection mtx from rotation mtx and translaction vec
def buildProjectionMatrix(rotationMtx, translationVector):
    # TODO: np.hstack could be used?
    translationVecAsColumn = np.array([[i] for i in translationVector])
    return np.concatenate((rotationMtx, translationVecAsColumn), axis=1)

# Draw small circle on an image
# `draw` is a ImageDraw from PIL
def drawSmallCircle(draw, x, y, colorPoint = 256, colorCircle = 200, width = 3):
    # Top left and bottom right corners
    draw.ellipse((x - width, y - width, x + width, y + width), fill = colorCircle)
    draw.point((x, y), fill = colorPoint)

# Make a filename, but with suffix appended before extension part
def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

# Convert from 3D homogenous coords
def from3Homogenous(coords):
    assert(coords.shape[1] == 4)
    return coords[:,0:3] / coords[:,3][:,None]

# Convert from 2D homogenous coords
def from2Homogenous(coords):
    assert(coords.shape[1] == 3)
    return coords[:,0:2] / coords[:,2][:,None]

# Convert to homogenous coords
def toHomogenous(coords):
    return np.hstack((coords, np.ones(coords.shape[0])[:,None]))
