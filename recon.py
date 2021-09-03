#!/usr/bin/env python
# coding: utf-8

# from typing import Counter
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.lib import imag
import imutils
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from utils import compute_sift

LAYOUT_DIM = 512
SHELVES = 3
# SCALE= 0.015625
SCALE = 0.0390625
K =  np.array([[600 ,           0.  ,        748/2.    ],
               [  0.         , 600  ,        1000/2.    ],
               [  0.         ,  0.           ,        1.      ]])


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

def rackAndBoxBBs(gray):
    
    gray = gray.astype(np.uint8)
    boxThresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]
    boxBB = getBoundingBoxes(boxThresh)

    shelfThresh = cv2.threshold(gray,10,155,cv2.THRESH_BINARY)[1]
    rackBB = getBoundingBoxes(shelfThresh)
    
    return rackBB, boxBB

def getBBForLabel(front_slice, top_slice):

    # find the bounding box for boxes in top view
    topRackBBox, topBoxesBBox = rackAndBoxBBs(top_slice)
    topBoxesBBox.sort()

    # find the bounding box for boxes in front view
    frontRackBBox, frontBoxesBBox = rackAndBoxBBs(front_slice)
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

def plotter3DOpen(boxBB, rackBB, type=1, show=True):
    geometries = []
    vertices = []
    print(len(boxBB))
    print(len(boxBB[0]))
    print(boxBB[0][0].shape)
    if type == 1:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]] )  # X Y Z   
                vertices.append(np.asarray(mesh_box.vertices))                      
                geometries.append(mesh_box)

                # pcd = o3d.geometry.PointCloud() 
                # pcd.points = mesh_box.vertices                             
                # geometries.append(pcd)

        for shelfBoxes in rackBB: # for racks
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5]/32, depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1]+box[5], box[2]] )  # X Y Z   
                vertices.append(np.asarray(mesh_box.vertices))                      
                geometries.append(mesh_box)


    if False:

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
        geometries.append(mesh_frame)

        o3d.visualization.draw_geometries(geometries)

    return vertices, geometries

def projectToImage(image, vertices, K):
    pts = []
    # implot = plt.imshow(image)

    for boxPoints in vertices:
        imagePts = K @ boxPoints.T
        imagePts[0,:] = imagePts[0,:] / imagePts[2,:]
        imagePts[1,:] = imagePts[1,:] / imagePts[2,:]
        imagePts = imagePts[:2,:].T

        pts.append(imagePts)
    # # print(pts)
        # plt.scatter(imagePts[:,0], imagePts[:,1], c='r', s=10)

    # plt.show()
    return pts   


def wrapper_func(index):
    rackBB = []
    boxBB = []
    freeSpace = []
    freeSpaceAboveBox = []

    frontlayout = np.load("./sdf_final/topLayouts/front00000%d.npy"%(index))
    toplayout = np.load("./sdf_final/topLayouts/top00000%d.npy"%(index))

    for i in range(SHELVES):
        
        topRackBBox, topBoxesBBox, frontRackBBox, frontBoxesBBox = getBBForLabel(frontlayout[i,:,:], toplayout[i,:,:])

        boxBoundingBoxes = calculate3DBB(topBoxesBBox, frontBoxesBBox)
        rackBoundingBoxes = calculate3DBB(topRackBBox, frontRackBBox)
        
        boxBB.append(boxBoundingBoxes)
        rackBB.append(rackBoundingBoxes)

    vertices, geometries = plotter3DOpen(boxBB, rackBB, 1, False )
    RGBimg = plt.imread('./sdf_final/Images/00000%d.jpg'%(index))
    imagePoints = projectToImage(RGBimg, vertices, K)

    imagePoints_list = []
    vertices_list = []

    for ii in imagePoints:
        for jj in ii:
            imagePoints_list.append([int(jj[0]), int(jj[1])])

    print("before pick")
    print(len(imagePoints_list))
    RGBimg = cv2.imread('./sdf_final/Images/00000%d.jpg'%(index))
    RGBimg = cv2.cvtColor(RGBimg, cv2.COLOR_BGR2RGB)
    kp, des = compute_sift(RGBimg, imagePoints_list)
    print("check")
    print(len(kp))
    for ii in vertices:
        for jj in ii:
            vertices_list.append([jj[0], jj[1], jj[2]])
    
    return kp, des, vertices_list, imagePoints_list, geometries

wrapper_func(0)
