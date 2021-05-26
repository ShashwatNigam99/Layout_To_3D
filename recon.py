#!/usr/bin/env python
# coding: utf-8

# from typing import Counter
import cv2
import numpy as np
import random
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

LAYOUT_DIM = 512
SHELVES = 4
SCALE= 0.015625

# plotting part
def get_cube(x,y,z, length, width, height):   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    sx = np.cos(Phi)*np.sin(Theta)
    sy = np.sin(Phi)*np.sin(Theta)
    sz = np.cos(Theta)/np.sqrt(2)
    
    return sx, sy, sz

def makeCube(ax, x,y,z,length, width, height, color, alpha):
    sx,sy,sz = get_cube(x,y,z, length, width, height)
    ax.plot_surface(sx*length + x, sy*height + y, sz*width + z, color = color, alpha = alpha)

def getBoundingBoxes(img):
    boundingBoxes = []
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour, thereby saving memory
    # contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours[1]:
        x,y,w,h = cv2.boundingRect(cntr) # og code 
        # x,y,w,h = cv2.minAreaRect(cntr)
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
        # multiply all of the above by appropriate scaling factor
      
        # length (front horizontal)
        L = min(wt, wf) 
        # width (how deep inside)
        W = ht
        # height
        H = hf

        # X is towards right
        X = min(xt, xf) - LAYOUT_DIM/2 
        # Y is downwards, camera is at the height equal to middle of the layout
        Y = yf - LAYOUT_DIM/2 
        # Z is towards the front
        Z = LAYOUT_DIM - yt - W        
        
        Boxes.append(SCALE*np.array([X, Y, Z, L, W, H]))
        #             0  1  2  3  4  5
    
    return Boxes

def calculate3DBBx(topBBox, frontBBox):
    Boxes = []

    for i in range(len(topBBox)):
        # length
        length = min(topBBox[i][2], frontBBox[i][2]) 
        # width
        width = topBBox[i][3]
        # height
        height = frontBBox[i][3]
        # x is towards right
        x = int(length/2) + max(topBBox[i][0], frontBBox[i][0])
        # y is upwards
        y = int(height/2) + frontBBox[i][1]
        # z is comming out of the image
        z = int(width/2) + topBBox[i][1]
        Boxes.append([x, y, z, length, width, height])
    
    return Boxes

def stackBB(boxes, rackBB):
    starting_height = 0
    #print("FreeSpace : ",freeSpaces)
    for i in range(len(boxes)):
        for j in range(len(boxes[i])):
            boxes[i][j][1] += starting_height

        for j in range(len(rackBB[i])):
            rackBB[i][j][1] += starting_height
            height_increase = rackBB[i][j][5]
        starting_height -= height_increase
    return boxes, rackBB

def plotter3DMat(boxBB, rackBB):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')

    scale = 512

    boxes_color = ['g', 'r', 'c', 'm', 'y', 'k' ]

    freeSpaceVolume = 0
    #for the free space
    for Boxes in rackBB:
        for box in Boxes:
            x,y,z,length, width, height = box
            freeSpaceVolume += length*width*height
            makeCube(ax, x,y,z,length, width, height, 'y', 0.05)

    occupiedByBoxes = 0
    #for the boxes in the scene
    for Boxes in boxBB:
        for box in Boxes:
            x,y,z,length, width, height = box
            occupiedByBoxes += length*width*height
            makeCube(ax, x,y,z,length, width, height, [random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)], 1)

    print(freeSpaceVolume*(0.015625**3) - occupiedByBoxes*(0.015625**3))

    ax.set_xlim(0,scale)
    ax.set_ylim(0,scale)
    ax.set_zlim(0,scale)
    ax.grid(False)
    plt.axis('off')
    ax.view_init(-90,-90)
    plt.show()


def plotter3DOpen(boxBB, rackBB, type=1):
    geometries = []
    if type == 1:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]], relative=False)  # X Y Z       relative  
                print(np.asarray(mesh_box.vertices))                             
                geometries.append(mesh_box)
    elif type == 2:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]])  # X Y Z 
                pcd = mesh_box.sample_points_uniformly(number_of_points = 500)   
                print(np.asarray(pcd.points))                                 
                geometries.append(pcd)
    elif type == 3:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]], relative=False)  # X Y Z       relative  
                pcd = o3d.geometry.PointCloud() 
                pcd.points = mesh_box.vertices                             
                geometries.append(pcd)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
    geometries.append(mesh_frame)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    rackBB = []
    boxBB = []
    freeSpace = []
    freeSpaceAboveBox = []

    for i in range(SHELVES):
        topRackBBox, topBoxesBBox, frontRackBBox, frontBoxesBBox = getBBForLabel('./samples/1/', '000000_'+str(i)+".png")

        boxBoundingBoxes = calculate3DBB(topBoxesBBox, frontBoxesBBox)
        rackBoundingBoxes = calculate3DBB(topRackBBox, frontRackBBox)
        
        boxBB.append(boxBoundingBoxes)
        rackBB.append(rackBoundingBoxes)

    plotter3DOpen(boxBB, rackBB, 3)




