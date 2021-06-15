import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from itertools import permutations as permute

from recon import wrapper_func
from utils import compute_sift
from scipy import io

NUM_MATCHES = 30
K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])
#SIFT MATCHES
VIZ= True
image_index = 2
Positions = [np.array([-7.082789897918701, -2.38836, 5.098252296447754]),
             np.array([-7.082789897918701, -1.60836, 5.79825]),
             np.array([-7.082789897918701, -2.38836, 5.098252296447754]),
             np.array([-7.082789897918701, -2.38836, 5.098252296447754]),
             np.array([-5.02279, -5.95836, 5.098252296447754])]

Rotations = [np.array([[-0.0000,  1.0000, -0.0000],
                       [-0.0000,  0.0000, -1.0000],
                       [-1.0000, -0.0000,  0.0000]]),
             np.array([[-0.0000,  1.0000, -0.0000],
                       [-0.0000,  0.0000, -1.0000],
                       [-1.0000, -0.0000,  0.0000]]),
             np.array([[0.1392, 0.9903, -0.0000],
                       [-0.0000, 0.0000, -1.0000],
                       [-0.9903, 0.1392,  0.0000]]),
             np.array([[-0.0663,  0.9978,  0.0000],
                       [-0.0000,  0.0000, -1.0000],
                       [-0.9978, -0.0663,  0.0000]]),
             np.array([[0.2470,  0.9690, -0.0000],
                       [-0.0000, -0.0000, -1.0000],
                       [-0.9690,  0.2470,  0.0000]])]

P1_T = np.array([-7.082789897918701, -6.078360557556152, 5.098252296447754] )
P1_R = np.array([90.0, 0.0, 90.0])
P1_R_matrix = np.array([[-0.0000,  1.0000, -0.0000],
                       [-0.0000,  0.0000, -1.0000],
                       [-1.0000, -0.0000,  0.0000]])

P2_T = Positions[image_index-2]
P2_R_matrix = Rotations[image_index-2]

T_change = (P1_T - P2_T)
# blender to open3D
T_change = np.array([[T_change[1]],
                     [-T_change[2]],
                     [-T_change[0]]])
dist_coeff = np.zeros((1,5))
R_change_matrix = P1_R_matrix.T.dot(P2_R_matrix)
R_change_vec, _ = cv2.Rodrigues(R_change_matrix)
R_change_vec = np.array([[R_change_vec[1]],
                     [-R_change_vec[2]],
                        [-R_change_vec[0]]])

print("Ground truth rotation")
R_change_vec = R_change_vec.reshape((3,1))
rotation_matrix, _ = cv2.Rodrigues(R_change_vec)
print(rotation_matrix)
print("Ground truth translation")
print(T_change.squeeze())


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

def read_images(i,j): 
    # RGBimg_original = cv2.imread('./blendSample_1/blendSample/%d.png'%(i))
    # RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
    # RGBimg_slanted = cv2.imread('./blendSample_1/blendSample/%d.png'%(j))
    # RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)

    RGBimg_original = cv2.imread('./blendSample_zoom/blendSample/%s.png'%(str(i).zfill(6)))
    RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
    RGBimg_slanted = cv2.imread('./blendSample_zoom/blendSample/%s.png'%(str(j).zfill(6)))
    RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)
    return RGBimg_original, RGBimg_slanted

# Get specified number of matches between both images using BF matcher(Brute force matcher)

def feature_matching_PNP(img1, img2, kp1, des1, kp2, des2, vertices_original, numMatches = 100, viz = False):
    
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)[:numMatches]
    img2_points = []
    obj_points = []
    dist_coeff = np.zeros((1,5))

    dict_analyser = {}
    for m in matches:
        img2_points.append([int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1])])
        obj_points.append(vertices_original[m.queryIdx])
        dict_analyser[tuple(vertices_original[m.queryIdx])] = (int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1]))

    img2_points = np.array(img2_points,dtype=np.float32)
    obj_points = np.array(obj_points,dtype=np.float32)

    # data = {'points3D': obj_points.T.astype('float64'), 'points2D': img2_points.squeeze().T.astype('float64')}
    # io.savemat('sift4.mat',data)
    
    # print("USAC_MAGSAC")
    # (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(obj_points, img2_points,\
    #                                                         K, dist_coeff, cv2.USAC_FAST)
    (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(obj_points, img2_points,\
                                                         K, dist_coeff, reprojectionError = 1.0, flags = cv2.SOLVEPNP_EPNP)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print("Rotation (SIFT)")
    print(rotation_matrix)
    print("Translation vector (SIFT)")
    print(translation_vector)

    
    rvec, tvec = cv2.solvePnPRefineLM(obj_points, img2_points, K, dist_coeff, rotation_vector, translation_vector)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("AFTER REFINING")
    print("Rotation (SIFT) refined")
    print(rotation_matrix)
    print("Translation (SIFT) refined")
    print(tvec)


    if VIZ:
        
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        
        fig = plt.figure(figsize=(16,16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    return [rotation_vector, translation_vector, rvec, tvec]



img1, img2 = read_images(1,image_index) # reading the images
kp1, des1, vertices_original, imagePoints_original = wrapper_func() # Make sure that the paths are the same in both files.
kp2, des2 = harris_corner_detector(img2)
# kp2, des2 = sift_compute(img2, True)

[rotation_vector, translation_vector, rvec, tvec] = feature_matching_PNP(img1, img2, kp1, des1, kp2, des2, vertices_original, NUM_MATCHES) 

projected_points_refined, _ = cv2.projectPoints(np.array(vertices_original), rvec, tvec, K, dist_coeff)
projected_points_unrefined, _ = cv2.projectPoints( np.array(vertices_original), rotation_vector, translation_vector, K, dist_coeff)

projected_points_GT, _ = cv2.projectPoints( np.array(vertices_original), R_change_vec, T_change, K, dist_coeff)


####### Visualisation
print(len(projected_points_unrefined))
if True :

    for i in projected_points_refined:
        # print(int(i[0][0]),int(i[0][1]))
        # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=2)
        cv2.drawMarker(img2, (int(i[0][0]),int(i[0][1])),(0,0,255), markerType=cv2.MARKER_STAR, 
        markerSize=4, thickness=1, line_type=cv2.LINE_AA)

    for i in projected_points_unrefined:
        # print(int(i[0][0]),int(i[0][1]))
        # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=2)
        cv2.drawMarker(img2, (int(i[0][0]),int(i[0][1])),(255,255,0), markerType=cv2.MARKER_STAR, 
        markerSize=3, thickness=1, line_type=cv2.LINE_AA)

    # for i in projected_points_GT:
    #     # print(int(i[0][0]),int(i[0][1]))
    #     # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(255, 255, 0), thickness=2)
    #     cv2.drawMarker(img2, (int(i[0][0]),int(i[0][1])),(255, 0, 0), markerType=cv2.MARKER_SQUARE , 
    #     markerSize=2, thickness=1, line_type=cv2.LINE_AA)

    # cv2.imwrite(str(image_index)+"_with_SIFT_PNP.png", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    plt.imshow(img2)
    plt.show()