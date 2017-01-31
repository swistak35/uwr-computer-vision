import os.path
import scipy as sc
import scipy.ndimage
from Drawing import Drawing
from OrientationAssignment import OrientationAssignment
from FeatureDescripting import FeatureDescripting
from matching import FeatureMatching

# TODO: Add timeit to measure steps

notreDameFileset = (
    ("data/Notre Dame/1_o.jpg", "data/Notre Dame/1_o-featuresmat.mat"),
    ("data/Notre Dame/2_o.jpg", "data/Notre Dame/2_o-featuresmat.mat"),
    "data/Notre Dame/921919841_a30df938f2_o_to_4191453057_c86028ce1f_o.mat")
fountainFileset = (
    ("data/fountain/0000.png", "data/fountain/0000-featuresmat.mat"),
    ("data/fountain/0001.png", "data/fountain/0001-featuresmat.mat"), None)
dudaFileset = (
    ("data/duda/img_20170130_162706.jpg", "data/duda/img_20170130_162706-featuresmat.mat"),
    ("data/duda/c3bxl_zweaywcbm.jpg", "data/duda/c3bxl_zweaywcbm-featuresmat.mat"), None)
mountRushmoreFileset = (
    ("data/Mount Rushmore/9021235130_7c2acd9554_o.jpg", "data/Mount Rushmore/9021235130_7c2acd9554_o-featuresmat.mat"),
    ("data/Mount Rushmore/9318872612_a255c874fb_o.jpg","data/Mount Rushmore/9318872612_a255c874fb_o-featuresmat.mat"), "data/Mount Rushmore/9021235130_7c2acd9554_o_to_9318872612_a255c874fb_o.mat")
episcopalGaudiFileset = (
    ("data/Episcopal Gaudi/4386465943_8cf9776378_o.jpg", "data/Episcopal Gaudi/4386465943_8cf9776378_o-featuresmat.mat"),
    ("data/Episcopal Gaudi/3743214471_1b5bbfda98_o.jpg", "data/Episcopal Gaudi/3743214471_1b5bbfda98_o-featuresmat.mat"),
    "data/Episcopal Gaudi/4386465943_8cf9776378_o_to_3743214471_1b5bbfda98_o.mat")

filesets = [
        # notreDameFileset,
        # fountainFileset,
        # dudaFileset,
        # mountRushmoreFileset,
        episcopalGaudiFileset,
    ]

def run():
    COMPUTE_ORIENTATION = False
    COMPUTE_DESCRIPTION = False
    COMPUTE_MATCHING = True
    for fileset in filesets:
        print("=== Files: (%s, %s)" % (fileset[0][0], fileset[1][0]))
        # Loading first image
        (imageFilename1, featureFilename1) = fileset[0]
        basePath1 = os.path.splitext(imageFilename1)[0]
        sourceImage1 = sc.ndimage.imread(imageFilename1, flatten = True)

        # Loading second image
        (imageFilename2, featureFilename2) = fileset[1]
        basePath2 = os.path.splitext(imageFilename2)[0]
        sourceImage2 = sc.ndimage.imread(imageFilename2, flatten = True)

        drawing = Drawing()

        # Orientation assignment of first image
        if COMPUTE_ORIENTATION:
            oa1 = OrientationAssignment()
            (featuresWithOrientation1, featuresWithOrientationToDraw1) = oa1.compute(sourceImage1, featureFilename1)
            drawing.drawFeaturesWithOrientations(imageFilename1, basePath1 + "-with-features.jpg", featuresWithOrientationToDraw1)

        # Orientation assignment of second image
        if COMPUTE_ORIENTATION:
            oa2 = OrientationAssignment()
            (featuresWithOrientation2, featuresWithOrientationToDraw2) = oa2.compute(sourceImage2, featureFilename2)
            drawing.drawFeaturesWithOrientations(imageFilename2, basePath2 + "-with-features.jpg", featuresWithOrientationToDraw2)

        # Descripting - 1st img
        featuresPath1 = basePath1 + "-features.mat"
        if COMPUTE_DESCRIPTION:
            fd1 = FeatureDescripting()
            featuresWithDescriptors1 = fd1.compute(sourceImage1, featuresWithOrientation1)
            fd1.saveFeatureDescriptors(featuresPath1, featuresWithDescriptors1)

        # Descripting - 2nd img
        featuresPath2 = basePath2 + "-features.mat"
        if COMPUTE_DESCRIPTION:
            fd2 = FeatureDescripting()
            featuresWithDescriptors2 = fd2.compute(sourceImage2, featuresWithOrientation2)
            fd2.saveFeatureDescriptors(featuresPath2, featuresWithDescriptors2)

        # Matching
        if COMPUTE_MATCHING:
            fm = FeatureMatching()
            fm.compute(fileset[0][0], fileset[1][0], featuresPath1, featuresPath2)
            fm.drawTopMatches(basePath1 + "-matches.jpg", amount = 60)
            if fileset[2]:
                fm.verify(fileset[2])

run()
