import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from itertools import permutations as permute

from recon import wrapper_func
from utils import compute_sift
from scipy import io


K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

def harris_corner_detector(img):

    operatedImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    operatedImage = np.float32(operatedImage)
    
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    dest = cv2.dilate(dest, None)

    indices = np.where(dest > 0.001 * dest.max())

    indices_list = []
    for i in range(len(indices[0])):
        indices_list.append([indices[1][i],indices[0][i]])

    kp, des = compute_sift(img, indices_list)

    return kp, des

def sift_compute(img, viz = False):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)

    if viz:
        img_1 = cv2.drawKeypoints(gray,keypoints_1,img)
        plt.imshow(img_1)
        plt.show()

    return keypoints_1, descriptors_1
# Get specified number of matches between both images using BF matcher(Brute force matcher)
def feature_matching_PNP(img1, img2, kp1, des1, kp2, des2, vertices_original, numMatches = 100, viz = False):
    
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)[:numMatches]
    print("Number of matches:", len(matches))
    img2_points = []
    obj_points = []
    dist_coeff = np.zeros((1,5))

    dict_analyser = {}
    for m in matches:
        img2_points.append([int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1])])
        obj_points.append(vertices_original[m.queryIdx])
        # print(tuple(vertices_original[m.queryIdx]))
        dict_analyser[tuple(vertices_original[m.queryIdx])] = (int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1]))

    print(dict_analyser)
    img2_points = np.array(img2_points,dtype=np.float32)
    obj_points = np.array(obj_points,dtype=np.float32)


    print(img2_points.shape)
    print(obj_points.shape)

    data = {'points3D': obj_points.T.astype('float64'), 'points2D': img2_points.squeeze().T.astype('float64')}
    io.savemat('sift4.mat',data)
    
    (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(obj_points, img2_points, K, dist_coeff)
    print("rotation (SIFT)")
    print(rotation_vector)
    print("translation vector (SIFT)")
    print(translation_vector)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print(rotation_matrix)
    # print("AFTER REFINING")
    # rvec, tvec = cv2.solvePnPRefineLM(obj_points, img2_points, K, dist_coeff, rotation_vector, translation_vector)
    # print(rvec)
    # print(tvec)
    # print(-np.matrix(rotation_matrix).T * np.matrix(translation_vector))
    if True:
        
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        
        fig = plt.figure(figsize=(16,16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

def read_images(i,j): 
    RGBimg_original = cv2.imread('./blendSample_1/blendSample/%d.png'%(i))
    RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
    RGBimg_slanted = cv2.imread('./blendSample_1/blendSample/%d.png'%(j))
    RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)
    return RGBimg_original, RGBimg_slanted

img1, img2 = read_images(1,4) # reading the images

kp1, des1, vertices_original = wrapper_func() # Make sure that the paths are the same in both files.
kp2, des2 = harris_corner_detector(img2)
# kp2, des2 = sift_compute(img2, True)

feature_matching_PNP(img1, img2, kp1, des1, kp2, des2, vertices_original, 100)

