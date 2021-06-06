import cv2
import numpy as np

def compute_sift(img,list_):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = []
    for i in list_:
        kp.append(cv2.KeyPoint(i[0],i[1],5))

    kp, des = sift.compute(img,kp)
    return kp, des  