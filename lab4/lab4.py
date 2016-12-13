import scipy as sc
import scipy.io
import scipy.misc
import scipy.ndimage
import numpy as np
import numpy.linalg
# import cv2
# import PIL as pil
import os.path
# import Image, ImageDraw

ALPHA = 0.05
WINDOW_SIZE = 5 # window will be windowSize*2 + 1

def mkPath(filename, suffix):
    basePath, extPath = os.path.splitext(filename)
    return (basePath + suffix + extPath)

def computeResponseValue(coords, imgGauss0, imgGauss1):
    (x, y) = coords
    secondMomentMatrix = np.zeros(4).reshape(2,2)
    for yd in range(-WINDOW_SIZE, WINDOW_SIZE+1):
        for xd in range(-WINDOW_SIZE, WINDOW_SIZE+1):
            ix = imgGauss0[y + yd][x + xd] # Are these used in correct order?
            iy = imgGauss1[y + yd][x + xd]
            secondMomentMatrix[0][0] += ix * ix
            secondMomentMatrix[0][1] += ix * iy
            secondMomentMatrix[1][0] += ix * iy
            secondMomentMatrix[1][1] += iy * iy
    # eigvals = np.linalg.eigvals(secondMomentMatrix)
    # print("Eigvals:")
    # print(eigvals)
    # response1 = eigvals[0] * eigvals[1] - ALPHA * np.power(eigvals[0] + eigvals[1], 2)
    response2 = np.linalg.det(secondMomentMatrix)- ALPHA * np.power(np.trace(secondMomentMatrix), 2)
    # print("Responses:")
    # print([response1, response2])
    return response2

filename = "data/Notre Dame/1_o.jpg"
image = scipy.ndimage.imread(filename, flatten = True) # loading in grey scale

imageGaussian0 = sc.ndimage.filters.gaussian_filter1d(image, 1.0, order = 1, axis = 0)
imageGaussian1 = sc.ndimage.filters.gaussian_filter1d(image, 1.0, order = 1, axis = 1)

sc.misc.imsave(mkPath(filename, "-gauss-derivative-0"), imageGaussian0)
sc.misc.imsave(mkPath(filename, "-gauss-derivative-1"), imageGaussian1)

result = np.zeros(image.shape)

for y in range(WINDOW_SIZE, image.shape[0] - WINDOW_SIZE):
    print(y)
    for x in range(WINDOW_SIZE, image.shape[1] - WINDOW_SIZE):
        result[y][x] = computeResponseValue((x, y), imageGaussian0, imageGaussian1)

maximumValue = np.amax(result)
sc.misc.imsave(mkPath(filename, "-heat-map1"), result)

result = result / maximumValue
sc.misc.imsave(mkPath(filename, "-heat-map2"), result)
