import numpy as np
import cv2

LAYOUT_DIM = 512
SHELVES = 4
SCALE = 0.0390625
K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])
DATA_PATH = './blendSample'

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

############ Rotates 180 deg around x axis
CV_2_BLENDER = np.eye(4)
CV_2_BLENDER[1, 1] *= -1
CV_2_BLENDER[2, 2] *= -1

dist_coeff = np.zeros((1, 5))