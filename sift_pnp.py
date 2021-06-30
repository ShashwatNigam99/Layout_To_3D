import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from itertools import permutations as permute
from config import *
from reconstruction import reconstruction
from utils import extract_pose_from_json, pose_2_transformation, harris_corner_detector_sift

NUM_MATCHES = 30
VIZ= True


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

    (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(obj_points, img2_points,\
                                                         K, dist_coeff, reprojectionError = 1.0, flags = cv2.USAC_MAGSAC)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    print("BEFORE REFINING")
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

    return [rotation_vector, translation_vector]

save_directory = DATA_PATH
save_directory = os.path.join(save_directory, "")

# Make sure that the paths are the same in both files.
kp1, des1, corners_list_1, image_points_1 = reconstruction(save_directory)

############ Rotates 180 deg around x axis
cv_2_blender_transform = np.eye(4)
cv_2_blender_transform[1, 1] *= -1
cv_2_blender_transform[2, 2] *= -1

################################### Frame number 1

frame_number_1 = 1
json_file_name_1 = "frame_%08d_CameraPose.json" % frame_number_1
json_file_name_1 = os.path.join(save_directory, json_file_name_1)
# Pose transforms points from camera to world in the blender coordinate system
pose = extract_pose_from_json(json_file_name_1).squeeze()
transformation = pose_2_transformation(pose)

#################################### Frame number 2

frame_number_2 = 4
json_file_name_2 = "frame_%08d_CameraPose.json" % frame_number_2
json_file_name_2 = os.path.join(save_directory, json_file_name_2)
# Pose transforms points from camera to world in the blender coordinate system
pose_2 = extract_pose_from_json(json_file_name_2).squeeze()
transformation_2 = pose_2_transformation(pose_2)


# Transforms from right to left
# From cv convert to blender frame
# Perform relative transform in blender frame 
# Convert from blender to cv frame
relative_transform = np.linalg.inv(cv_2_blender_transform) @ np.linalg.inv(transformation_2) @ transformation @ cv_2_blender_transform

# Read both the images
frame_1 = cv2.cvtColor(cv2.imread(save_directory+str(frame_number_1).zfill(6)+".png"),cv2.COLOR_BGR2RGB)
frame_2 = cv2.cvtColor(cv2.imread(save_directory+str(frame_number_2).zfill(6)+".png"),cv2.COLOR_BGR2RGB)

# Detect corners in frame 2 and compute SIFT descriptors over them
kp2, des2 = harris_corner_detector_sift(frame_2)

[rotation_vector, translation_vector] = feature_matching_PNP(frame_1, frame_2, kp1, des1, kp2, des2,\
                                                                            corners_list_1, NUM_MATCHES) 

dist_coeff = np.zeros((1, 5))

# projected_points_refined, _ = cv2.projectPoints(np.array(corners_list_1), rvec, tvec, K, dist_coeff)
projected_points_unrefined, _ = cv2.projectPoints( np.array(corners_list_1), rotation_vector, translation_vector, K, dist_coeff)


Rotation = relative_transform[:3, :3]
Translation = relative_transform[:3, -1]
print("Rotation (GT)")
print(Rotation)
print("Translation vector (GT)")
print(Translation)
projected_points_GT, _ = cv2.projectPoints(np.array(corners_list_1), Rotation, Translation, K, dist_coeff)

if True :

    for i in projected_points_unrefined:
        cv2.drawMarker(frame_2, (int(i[0][0]),int(i[0][1])),(255,255,0), markerType=cv2.MARKER_STAR, 
        markerSize=3, thickness=1, line_type=cv2.LINE_AA)

    for i in projected_points_GT:
        cv2.drawMarker(frame_2, (int(i[0][0]),int(i[0][1])),(255, 0, 0), markerType=cv2.MARKER_SQUARE , 
        markerSize=2, thickness=1, line_type=cv2.LINE_AA)