import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from itertools import permutations as permute
from config import *
from reconstruction import reconstruction
from utils import get_image_and_transformation, get_relative_tranformation_cv, harris_corner_detector_sift

NUM_MATCHES = 50
VIZ= True
FRAMES = 60

# Get specified number of matches between both images using BF matcher(Brute force matcher)

def feature_matching_PNP(img1, img2, kp1, des1, kp2, des2, vertices_original, numMatches = 100, viz = False):
    
    # feature matching
    # !can mess around with distance function
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # query, train
    matches = bf.match(des1, des2)
    # take the best n matches, n=numMatches
    matches = sorted(matches, key = lambda x:x.distance)[:numMatches]
    
    img2_points = []
    obj_points = []
    dist_coeff = np.zeros((1,5))

    for m in matches:
        img2_points.append([int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1])])
        obj_points.append(vertices_original[m.queryIdx])

    img2_points = np.array(img2_points,dtype=np.float32)
    obj_points = np.array(obj_points,dtype=np.float32)

    (_, rotation_vector, translation_vector, _) = cv2.solvePnPRansac(obj_points, img2_points,\
                                                 K, dist_coeff, reprojectionError = 1.0, flags = cv2.USAC_MAGSAC)
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print("BEFORE REFINING")
    print("Rotation (SIFT)")
    print(rotation_matrix)
    print("Translation vector (SIFT)")
    print(translation_vector)

    rotation_vector_refined, translation_vector_refined = cv2.solvePnPRefineLM(obj_points, img2_points, K, dist_coeff, rotation_vector, translation_vector)
    rotation_matrix_refined, _ = cv2.Rodrigues(rotation_vector_refined)
    print("AFTER REFINING")
    print("Rotation (SIFT) refined")
    print(rotation_matrix_refined)
    print("Translation (SIFT) refined")
    print(translation_vector_refined)

    if VIZ:
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)
        fig = plt.figure(figsize=(16,16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    return [rotation_vector, translation_vector, \
            rotation_vector_refined, translation_vector_refined,\
            matches]


# get all frames and associated transformations
images, transformations = get_image_and_transformation(DATA_PATH, FRAMES, list= True)

# Take first frame and reconstruct it
kp1, des1, vertices_1, imgpts_1 = reconstruction(DATA_PATH)

vertices_1 = np.array(vertices_1, dtype=np.float32)
imgpts_1 = np.array(imgpts_1, dtype=np.float32)

print(vertices_1.shape)

for i in range(2,FRAMES+1):
    # matching frame 1(i-2) and frame 2(i-1)

    # get ground truth relative transform between the two frames 
    relative_transform = get_relative_tranformation_cv(transformations[i-2],transformations[i-1])

    # Detect corners in frame 2 and compute SIFT descriptors over them
    kp2, des2 = harris_corner_detector_sift(images[i-1])

    [rotation_vector, translation_vector, rotation_vector_refined, translation_vector_refined, matches] \
    = feature_matching_PNP(images[i-2],images[i-1], kp1, des1, kp2, des2, vertices_1, NUM_MATCHES) 

    projected_points_refined, _ = cv2.projectPoints(np.array(vertices_1), rotation_vector_refined, translation_vector_refined, K, dist_coeff)
    projected_points_unrefined, _ = cv2.projectPoints( np.array(vertices_1), rotation_vector, translation_vector, K, dist_coeff)

    Rotation = relative_transform[:3, :3]
    Translation = relative_transform[:3, -1]
    print("Rotation (GT)")
    print(Rotation)
    print("Translation vector (GT)")
    print(Translation)
    projected_points_GT, _ = cv2.projectPoints(np.array(vertices_1), Rotation, Translation, K, dist_coeff)

    if True :

        for pt in projected_points_unrefined:
            cv2.drawMarker(images[i-1], (int(pt[0][0]),int(pt[0][1])),(255,255,0), markerType=cv2.MARKER_STAR, 
            markerSize=3, thickness=1, line_type=cv2.LINE_AA)

        for pt in projected_points_GT:
            cv2.drawMarker(images[i-1], (int(pt[0][0]),int(pt[0][1])),(255, 0, 0), markerType=cv2.MARKER_SQUARE , 
            markerSize=2, thickness=1, line_type=cv2.LINE_AA)
        
        plt.axis('off')
        plt.imshow(images[i-1])
        plt.show()

    # carry over the matches for the next frame
    kp1 = np.array([kp2[m.trainIdx] for m in matches])
    des1 = np.array([des2[m.trainIdx] for m in matches])
    vertices_1 = np.array([vertices_1[m.queryIdx] for m in matches])
    print(vertices_1.shape)
    vertices_1 = Rotation @ vertices_1.T + Translation.reshape((3,1))
    vertices_1 = vertices_1.T