from numpy.core.arrayprint import printoptions
from scipy.spatial.transform.rotation import Rotation
from recon import wrapper_func
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])
dist_coeff = np.zeros((1,5))

test_dict = {(5.8203125, -3.3984375, 12.3828125): (272, 71), (4.8046875, -4.0234375, 12.3828125): (248, 54), (1.328125, -4.609375, 13.4375): (168, 51), (0.3125, -3.515625, 12.3828125): (143, 67), (-0.078125, 1.8359375, 12.3828125): (134, 193), (4.8046875, -4.53125, 12.3828125): (248, 45), (-0.078125, 1.8359375, 13.3984375): (134, 191), (4.8046875, -4.53125, 13.3984375): (246, 52), (1.328125, -4.0234375, 12.34375): (170, 54), (8.828125, -4.0234375, 12.1875): (343, 54), (8.0078125, 1.8359375, 12.1875): (325, 196), (-1.796875, 4.765625, 13.3984375): (102, 254), (3.7890625, -4.53125, 12.3828125): (225, 45), (-1.796875, 4.1796875, 13.3984375): (102, 242), (7.109375, 1.8359375, 13.59375): (297, 191), (6.1328125, -2.734375, 12.3828125): (281, 88)}

for k,v in test_dict.items():
    print(str(k) + ": " + str(v))

kp1, des1, corners_list = wrapper_func() # Make sure that the paths are the same in both files.

K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

image_index = 3
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

corners_list = np.array(corners_list)

dist_coeff = np.zeros((1,5))

T_change = -(P2_T - P1_T)
T_change = np.array([[T_change[1]],
                     [-T_change[2]],
                        [-T_change[0]]])

R_change_matrix = P1_R_matrix.T.dot(P2_R_matrix)
R_change_vec, _ = cv2.Rodrigues(R_change_matrix)
R_change_vec = np.array([[R_change_vec[1]],
                     [-R_change_vec[2]],
                        [-R_change_vec[0]]])

corners_array = np.array(corners_list)
corners_2d = list(test_dict.keys())
corners_2d = [list(elem) for elem in corners_2d]
corners_2d = np.array(corners_2d)   
# print(corners_2d)

projected_points, _ = cv2.projectPoints( corners_list, R_change_vec, T_change, K, dist_coeff)

corners_array_half = []
projected_points_half = []
dict_2 = {}
for j in corners_2d:
    for count, i in enumerate(corners_array):
 
        if i[0] == j[0] and i[1] == j[1] and i[2] == j[2]:
            
            corners_array_half.append(i)
            projected_points_half.append(projected_points[count])
            dict_2[tuple(i)] = projected_points[count]

# print(test_dict)
# print(dict_2)
for k,v in dict_2.items():
    print(str(k) + ": " + str(tuple(v[0])))

corners_array_half = np.array(corners_array_half)
projected_points_half = np.array(projected_points_half)
(_, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(corners_array_half, projected_points_half, K, dist_coeff)

print("Rotation from PnP (Ground Truth)")
print(rotation_vector)
print("Translation from PnP (Ground Truth)")
print(translation_vector)
