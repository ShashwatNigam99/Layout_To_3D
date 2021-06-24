import os
from numpy.lib.npyio import save
import cv2
import numpy as np
import matplotlib.pyplot as plt
from reconstruction import reconstruction
from config import *
from utils import extract_pose_from_json, pose_2_transformation

FLOAT_EPS = np.finfo(np.float).eps

save_directory = DATA_PATH
save_directory = os.path.join(save_directory, "")

kp1, des1, corners_list, _ = reconstruction(save_directory)
dist_coeff = np.zeros((1, 5))

############ Rotates 180 deg around x axis
cv_2_blender_transform = np.eye(4)
cv_2_blender_transform[1, 1] *= -1
cv_2_blender_transform[2, 2] *= -1

#################################### Frame number 1

frame_number_1 = 1
json_file_name_1 = "frame_%08d_CameraPose.json" % frame_number_1
json_file_name_1 = os.path.join(save_directory, json_file_name_1)

pose_1 = extract_pose_from_json(json_file_name_1).squeeze()
transformation = pose_2_transformation(pose_2)

#################################### Frame number 2

frame_number_2 = 79
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

frame_2 = cv2.imread(save_directory+str(frame_number_2).zfill(6)+".png")
frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)

Rotation = relative_transform[:3, :3]
Translation = relative_transform[:3, -1]

dist_coeff = np.zeros((1, 5))
corners_list = np.array(corners_list)
print(corners_list.shape)
projected_points, _ = cv2.projectPoints(corners_list, Rotation, Translation, K, dist_coeff)

current_frame = cv2.imread( save_directory+str(frame_number_1).zfill(6)+".png")
current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

corners_3d = np.array(corners_list)

plt.imshow(frame_2)

for i in projected_points:
    plt.scatter(i[0][0], i[0][1])
plt.show()
