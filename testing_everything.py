import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

obj_points= np.random.uniform(low=100, high=200, size=(160,3))
dist_coeff = np.zeros((1,5))

rot_angles =np.random.uniform(low=0, high=359, size=(1,3))
rot_vec = R.from_euler('xyx', rot_angles, degrees=True).as_rotvec()
t_vec = np.random.uniform(low=5, high=20, size=(1,3))

print("Ground Truth rotation vector")
print(rot_vec.T)
print("Ground Truth translation vector")
print(t_vec.T)

img2_points, _ = cv2.projectPoints(obj_points, rot_vec, t_vec, K, dist_coeff)
# print(img2_points)

(_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(obj_points, img2_points, K, dist_coeff)

print("Predicted rotation vector")
print(rotation_vector)
print("Predicted Translation vector")
print(translation_vector)