# Testing

# from sift import R_change, T_change
from numpy.core.arrayprint import printoptions
from recon import wrapper_func
import numpy as np
import cv2
import matplotlib.pyplot as plt

imagePoints_original, corners_list = wrapper_func() # Make sure that the paths are the same in both files.

K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

P1_T = np.array([-7.082789897918701, -6.078360557556152, 5.098252296447754] )
P1_R = np.array([90, 0, 90])

P2_T = np.array([-7.082789897918701, -2.38836, 5.098252296447754] )
P2_R = np.array([90, 0, 90])

corners_list = np.array(corners_list)

dist_coeff = np.zeros((1,5))

# def get_RT_matrix(R_change,T_change):
#     R_matrix = R.from_euler('xyz', R_change, degrees=True)
#     R_matrix = np.array(R_matrix.as_matrix())
#     RT_matrix = np.array([[0.0 for i in range(4)] for j in range(3)])
#     for i in range(3):
#         for j in range(4):
#             if(j!=3):
#                 RT_matrix[i,j] = R_matrix[i,j]
#             else:
#                 RT_matrix[i,j] = T_change[i]
#     return RT_matrix

T_change = P2_T - P1_T
T_change /= 2.5
T_change = np.array([[T_change[1]],
                        [-T_change[2]],
                        [-T_change[0]]])

R_change = np.array([[0.0],
                     [0.0], 
                     [0.0]])

projected_points, _ = cv2.projectPoints( corners_list, R_change, T_change, K, dist_coeff)

RGBimg_original = cv2.imread('./blendSample_1/blendSample/1.png')
RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
RGBimg_slanted = cv2.imread('./blendSample_1/blendSample/2.png')
RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)

img1 = RGBimg_original
img2 = RGBimg_slanted

# Plot the points on image
for i in projected_points:
    print(int(i[0][0]),int(i[0][1]))
    img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=3)

plt.imshow(img2)
plt.xlim([0, 600])
plt.ylim([0, 600])
plt.show()