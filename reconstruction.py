import cv2, imutils, sys, os
import numpy as np
import matplotlib.pyplot as plt

from config import *
from utils import compute_sift, plotter3DOpen, projectToImage

def getBoundingBoxes(img):
    boundingBoxes = []
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby saving memory
    # contours = contours[0] if len(contours) == 2 else contours[1]
    contours = contours[1] if imutils.is_cv3() else contours[0]
    # print(contours[1].type)
    # contours[1] = np.array(contours[1], dtype = np.float64)
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr) # og code 
        # x,y,w,h = cv2.minAreapltRect(cntr)
        # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html 
        boundingBoxes.append([x,y,w,h])
    return boundingBoxes

def rackAndBoxBBs(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    boxThresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]
    boxBB = getBoundingBoxes(boxThresh)

    shelfThresh = cv2.threshold(gray,10,155,cv2.THRESH_BINARY)[1]
    rackBB = getBoundingBoxes(shelfThresh)
    
    return rackBB, boxBB

def getBBForLabel(basePath, layoutPath):

    # find the bounding box for boxes in top view
    img = cv2.imread(basePath + 'top' + layoutPath)
    topRackBBox, topBoxesBBox = rackAndBoxBBs(img)
    topBoxesBBox.sort()

    # find the bounding box for boxes in front view
    img = cv2.imread(basePath + 'front' + layoutPath)
    frontRackBBox, frontBoxesBBox = rackAndBoxBBs(img)
    frontBoxesBBox.sort()

    return topRackBBox, topBoxesBBox, frontRackBBox, frontBoxesBBox

def calculate3DBB(topBBox, frontBBox):
    Boxes = []
    for i in range(len(topBBox)):
        # print("x,y", topBBox[i][0], topBBox[i][1], "<-->", frontBBox[i][0], frontBBox[i][1])
        # print("w,h", topBBox[i][2], topBBox[i][3], "<-->", frontBBox[i][2], frontBBox[i][3])
      
        xt, yt, wt, ht = topBBox[i][0], topBBox[i][1], topBBox[i][2], topBBox[i][3]
        xf, yf, wf, hf = frontBBox[i][0], frontBBox[i][1], frontBBox[i][2], frontBBox[i][3] 
        # length (front horizontal)
        L = min(wt, wf) 
        # width (how deep inside)
        W = ht
        # height
        # X is towards right
        H = hf
        X = min(xt, xf) - LAYOUT_DIM/2 
        # Y is downwards, camera is at the height equal to middle of the layout
        Y = yf - LAYOUT_DIM/2 
        # Z is towards the front
        Z = LAYOUT_DIM - yt - W        
        # multiply all of the above by appropriate scaling factor
        Boxes.append(SCALE*np.array([X, Y, Z, L, W, H]))
        #                            0  1  2  3  4  5
    return Boxes

def reconstruction(dir_path = None):
    # Reconstructs image 1 (000001.png) in specified directory
    rackBB = []
    boxBB = []
    freeSpace = []
    freeSpaceAboveBox = []

    # Ensure directory is specified and valid paths exist to all shelves
    if dir_path is None:
        sys.exit("No directory path specified in function reconstruction")
    for i in range(SHELVES):
        try:
            dir_path = os.path.join(dir_path, "")
            topRackBBox, topBoxesBBox, frontRackBBox, frontBoxesBBox = getBBForLabel(
                                                                        dir_path, '000001_'+str(i)+".png")
        except OSError as e:
            sys.exit(e)
 
        boxBoundingBoxes = calculate3DBB(topBoxesBBox, frontBoxesBBox)
        rackBoundingBoxes = calculate3DBB(topRackBBox, frontRackBBox)
        
        boxBB.append(boxBoundingBoxes)
        rackBB.append(rackBoundingBoxes)

    vertices = plotter3DOpen(boxBB, rackBB, 1, False )

    RGBimg = plt.imread(dir_path + '000001.png')

    imagePoints = projectToImage(RGBimg, vertices, K)

    imagePoints_list = []
    vertices_list = []

    for ii in imagePoints:
        for jj in ii:
            imagePoints_list.append([int(jj[0]), int(jj[1])])

    RGBimg = cv2.cvtColor(cv2.imread(dir_path + '000001.png'), cv2.COLOR_BGR2RGB)

    kp, des = compute_sift(RGBimg, imagePoints_list)
    for ii in vertices:
        for jj in ii:
            vertices_list.append([jj[0], jj[1], jj[2]])
    
    return kp, des, vertices_list, imagePoints_list

