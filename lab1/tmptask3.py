
import numpy as np

import scipy.ndimage
image = scipy.ndimage.imread("data/task34/CalibIm1.gif")

import numpy as np
cx,cy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))

r = np.stack((cx,cy), axis=2).reshape((-1,2), order='F')

from scipy import ndimage
# ndimage.map_coordinates(a, [[0.5, 2, 3], [0.5, 1, 2]], order=1)

imager = image[:,:,0]
imageg = image[:,:,1]
imageb = image[:,:,2]

from task3 import *
distortion, intrinsicMtx, mtxs = loadCalibData("data/task34/Calib.txt")
extrinsicMatrix = mtxs[0]

endhomor = np.array([ (p[0], p[1], 1.0) for p in r ])
intrInv = np.linalg.inv(intrinsicMtx)
normalizedHomopoints = intrInv.dot(endhomor.transpose())
projectedPoints = np.array([ (hp[0] / hp[2], hp[1] / hp[2]) for hp in normalizedHomopoints.transpose() ])
correctedPoints = np.array([ correctedPoint(p, distortion) for p in projectedPoints ])
homopoints2 = np.array([ (p[0], p[1], 1.0) for p in correctedPoints ])
points3 = np.array([ intrinsicMtx.dot(p) for p in homopoints2 ])
points4 = np.array([ (hp[0] / hp[2], hp[1] / hp[2]) for hp in points3 ])
mappedPointsR = ndimage.map_coordinates(imager, points4.transpose(), order=3).reshape(480, 640)
mappedPointsG = ndimage.map_coordinates(imageg, points4.transpose(), order=3).reshape(480, 640)
mappedPointsB = ndimage.map_coordinates(imageb, points4.transpose(), order=3).reshape(480, 640)
newimage = np.stack((mappedPointsR, mappedPointsG, mappedPointsB), axis=-1)
scipy.misc.imsave("data/task34/bar.gif", newimage)

distortion, intrinsicMtx, mtxs = loadCalibData("data/task34/Calib.txt")
points = loadModelPoints("data/task34/Model.txt")
homopoints = [ (p[0], p[1], 0.0, 1.0) for p in points ]

# Undistorted pictures
savePictureWithPoints("data/task34/bar.gif", intrinsicMtx, mtxs[0], homopoints)
