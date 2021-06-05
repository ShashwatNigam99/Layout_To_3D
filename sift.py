import cv2
import numpy as np
import matplotlib.pyplot as plt
from recon import wrapper_func
from scipy.spatial.transform import Rotation as R
from itertools import permutations as permute

K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

def display_on_image(img,points_list):

    points_list = np.array(points_list)
    plt.imshow(img)
    plt.scatter(points_list[:,0], points_list[:,1], c='r', s=10)
    plt.show()

def harris_corner_detector(img):

    operatedImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    operatedImage = np.float32(operatedImage)
    
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    dest = cv2.dilate(dest, None)

    indices = np.where(dest > 0.001 * dest.max())

    indices_list = []
    for i in range(len(indices[0])):
        indices_list.append([indices[1][i],indices[0][i]])

    return indices_list

# Get specified number of matches between both images using BF matcher(Brute force matcher)
def feature_matching(img1, img2, kp1, des1, kp2, des2, list1, list2, vertices_original, numMatches = 50, viz = False):
    
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)[:numMatches]

    img2_points = []
    obj_points = []
    dist_coeff = np.zeros((1,5))

    for m in matches:
        img2_points.append(list2[m.trainIdx])
        obj_points.append(vertices_original[m.queryIdx])


    img2_points = np.array(img2_points,dtype=np.float32)
    obj_points = np.array(obj_points,dtype=np.float32)


    print(img2_points.shape)
    print(obj_points.shape)
    
    (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(obj_points, img2_points, K, dist_coeff)
    print(rotation_vector)
    print(translation_vector)

    if True:
        
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        
        fig = plt.figure(figsize=(16,16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

def compute_sift(img1,list1,img2,list2,vertices_original,thresh = 0.75):

    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = []
    kp2 = []

    for i in list1:
        kp1.append(cv2.KeyPoint(i[0],i[1],3))
    for i in list2:
        kp2.append(cv2.KeyPoint(i[0],i[1],3))

    kp1, des1 = sift.compute(img1,kp1)
    kp2, des2 = sift.compute(img2,kp2)

    feature_matching(img1, img2, kp1, des1, kp2, des2, list1, list2, vertices_original)

   
RGBimg_original = cv2.imread('./blendSample_1/blendSample/1.png')
RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
RGBimg_slanted = cv2.imread('./blendSample_1/blendSample/2.png')
RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)

imagePoints_original, vertices_original = wrapper_func() # Make sure that the paths are the same in both files.
images_points_slanted = harris_corner_detector(RGBimg_slanted)

compute_sift(RGBimg_original, imagePoints_original, RGBimg_slanted, images_points_slanted,vertices_original, 0.75)
# display_on_image(RGBimg_original,imagePoints_original)
# display_on_image(RGBimg_slanted,images_points_slanted)

# print(R.from_euler('XYZ', [np.pi/2, 0, np.pi/2]).as_matrix())