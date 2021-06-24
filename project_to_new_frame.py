import os
from numpy.lib.npyio import save
import cv2
import numpy as np
import matplotlib.pyplot as plt
from reconstruction import reconstruction
from config import *
from utils import get_image_and_transformation, get_relative_tranformation_cv

FLOAT_EPS = np.finfo(np.float).eps

save_directory = DATA_PATH
save_directory = os.path.join(save_directory, "")

kp1, des1, corners_list, _ = reconstruction(save_directory)

#################################### Frame number 1

frame_number_1 = 1
frame_1, transformation_1 = get_image_and_transformation(save_directory, frame_number_1)

#################################### Frame number 2

frame_number_2 = 79
frame_2, transformation_2 = get_image_and_transformation(save_directory, frame_number_2)

relative_transform = get_relative_tranformation_cv(transformation_1, transformation_2)

Rotation = relative_transform[:3, :3]
Translation = relative_transform[:3, -1]

corners_list = np.array(corners_list)
print(corners_list.shape)
projected_points, _ = cv2.projectPoints(corners_list, Rotation, Translation, K, dist_coeff)

corners_3d = np.array(corners_list)

plt.imshow(frame_2)

for i in projected_points:
    plt.scatter(i[0][0], i[0][1])
plt.show()
