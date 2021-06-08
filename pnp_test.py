# Testing

from numpy.core.arrayprint import printoptions
from scipy.spatial.transform.rotation import Rotation
from recon import wrapper_func
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import io

kp1, des1, corners_list = wrapper_func() # Make sure that the paths are the same in both files.
viz = False
K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

image_index = 6
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

T_change = (P1_T - P2_T)
# blender to open3D
T_change = np.array([[T_change[1]],
                     [-T_change[2]],
                     [-T_change[0]]])

R_change_matrix = P1_R_matrix.T.dot(P2_R_matrix)
R_change_vec, _ = cv2.Rodrigues(R_change_matrix)
R_change_vec = np.array([[R_change_vec[1]],
                     [-R_change_vec[2]],
                        [-R_change_vec[0]]])

corners_array = np.array(corners_list)

projected_points, _ = cv2.projectPoints( corners_list, R_change_vec, T_change, K, dist_coeff)

print(corners_array.T.shape,  projected_points.squeeze().T.shape)
data = {'points3D': corners_array.T, 'points2D': projected_points.squeeze().T}
io.savemat('test4.mat',data)

(_, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(corners_array, projected_points, K, dist_coeff)
projected_points_PnP, _ = cv2.projectPoints( corners_list, rotation_vector, translation_vector, K, dist_coeff)


print("Ground truth translation")
print(T_change.squeeze())
print("PnP Translations :\n", translation_vector.squeeze())

print("Ground truth rotation")
R_change_vec = R_change_vec.reshape((3,1))
rotation_matrix, _ = cv2.Rodrigues(R_change_vec)
print(rotation_matrix)
print("PnP Rotations : \n", cv2.Rodrigues(rotation_vector.squeeze())[0])

if viz:
    RGBimg_original = cv2.imread('./blendSample_1/blendSample/1.png')
    RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
    RGBimg_slanted = cv2.imread('./blendSample_1/blendSample/'+str(image_index)+'.png')
    RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)

    img1 = RGBimg_original
    img2 = RGBimg_slanted


    for i in projected_points:
        # print(int(i[0][0]),int(i[0][1]))
        # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=2)
        cv2.drawMarker(img2, (int(i[0][0]),int(i[0][1])),(0,0,255), markerType=cv2.MARKER_STAR, 
        markerSize=4, thickness=2, line_type=cv2.LINE_AA)


    for i in projected_points_PnP:
        # print(int(i[0][0]),int(i[0][1]))
        # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(255, 255, 0), thickness=2)
        cv2.drawMarker(img2, (int(i[0][0]),int(i[0][1])),(255, 255, 0), markerType=cv2.MARKER_SQUARE , 
        markerSize=2, thickness=1, line_type=cv2.LINE_AA)

    cv2.imwrite(str(image_index)+"_with_GT_PNP.png", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    plt.imshow(img2)
    plt.show()