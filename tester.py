from scipy.spatial.transform.rotation import Rotation
from recon import wrapper_func
import numpy as np
import cv2
import matplotlib.pyplot as plt

import scipy.io

mat = scipy.io.loadmat('./Matlab Stuff/sift4.mat')
print(mat["points3D"].T)
print(mat["points2D"].T)