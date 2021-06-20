from numpy.core.arrayprint import printoptions
from scipy.spatial.transform.rotation import Rotation
from recon import wrapper_func
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import io
import re

kp1, des1, corners_list, _ = wrapper_func() # Make sure that the paths are the same in both files.
# print(corners_list)
viz = False
dist_coeff = np.zeros((1,5))
K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])


def get_RT_matrix_from_file(file_path):
    file = open(file_path)
    lines = file.read().replace("\n", " ")
    res = re.findall(r'\<Matrix.*?\>', lines)
    R_M = []
    T_M = []
    for i in range(len(res)):
        R_Matrix = []
        T_Matrix = []
        matrix = res[i]
        rows = re.findall(r'\(.*?\)', matrix)
        for a in rows:
            row = a.split(",")
            row[0] = row[0][1:]
            row[-1] = row[-1][:-1]
            r = []
            t = []
            tm = row[-1]
            row = row[ : -1]
            for m in row:
                r.append(float(m))
            R_Matrix.append(r)
            T_Matrix.append(float(tm))
        R_M.append(np.array(R_Matrix))
        T_M.append(np.array(T_Matrix))
    return R_M, T_M

def get_vector_from_file(file_path):
    file = open(file_path)
    lines = file.read().replace("\n", " ")
    res = re.findall(r'\<Vector.*?\>', lines)
    T_M = []
    for i in range(len(res)):
        T_Matrix = []
        matrix = res[i]
        rows = re.findall(r'\(.*?\)', matrix)
        for a in rows:
            row = a.split(",")
            row[0] = row[0][1:]
            row[-1] = row[-1][:-1]
            t = []
            for j in row:
                t.append(float(j))
            T_Matrix.append(t)
        T_M.append(np.array(T_Matrix))
    return T_M

def get_Relative_matrix_list(list_R_matrix, list_T_matrix):
    i = 1;
    list_of_relative_R_matrix = []
    list_of_relative_T_matrix = []
    while(i < len(list_R_matrix)):
        R_previous = list_R_matrix[i-1]
        R_current = list_R_matrix[i]
        relative_R = R_previous.T.dot(R_current) # fill the formula here
        list_of_relative_R_matrix.append(relative_R)

        T_previous = list_T_matrix[i-1]
        T_current = list_T_matrix[i]
        # print(T_previous, T_current)
        relative_T = -(T_previous - T_current)
        list_of_relative_T_matrix.append(relative_T)
        i += 1
    return list_of_relative_R_matrix, list_of_relative_T_matrix

def get_Relative_R_matix(R_previous, R_current):
    relative_R = R_previous.T.dot(R_current) # fill the formula here
    return relative_R

def get_Relative_T(T_previous, T_current):
    relative_T = -(T_previous - T_current)
    return relative_T

# R_Matrix, T_Matrix = get_RT_matrix_from_file("./camera_positions_and_angles.txt")
R_Matrix, T_Matrix = get_RT_matrix_from_file("./blendSample_video/blendSample/camera_positions_and_angles.txt")
R_Relative, T_Relative = get_Relative_matrix_list(R_Matrix, T_Matrix)

# define which images you want to get the R and T for
previous_frame_number = 1
current_frame_number = 15

previous_frame_path = "./blendSample_video/blendSample/"+str(previous_frame_number).zfill(6)+".png"
current_frame_path = "./blendSample_video/blendSample/"+str(current_frame_number).zfill(6)+".png"

previous_frame = cv2.imread(previous_frame_path)
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2RGB)
current_frame = cv2.imread(current_frame_path)
current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

previous_frame_R = R_Matrix[previous_frame_number - 1]
current_frame_R = R_Matrix[current_frame_number - 1]

# print(current_frame_R)

previous_frame_T = T_Matrix[previous_frame_number - 1]
current_frame_T = T_Matrix[current_frame_number - 1]


# T = get_vector_from_file("./blendSample_video/blendSample/camera_positions_and_angles.txt")
# for i in T:
#     print(T)
#     print()




# compute the relative motion
Relative_R_matix = get_Relative_R_matix(previous_frame_R, current_frame_R)
Relative_T_matix = get_Relative_T(previous_frame_T, current_frame_T)


# transform to open3d coordinates
# Relative_R_matix, _ = cv2.Rodrigues(Relative_R_matix)
# print("Before : ", Relative_R_matix)
# Relative_R_matix = np.array([[Relative_R_matix[1]],
#                      [-Relative_R_matix[2]],
#                         [-Relative_R_matix[0]]])

# Relative_T_matix = np.array([[Relative_T_matix[1]],
#                      [-Relative_T_matix[2]],
#                      [-Relative_T_matix[0]]])

# print("After : ", Relative_R_matix)

# get the projected points
corners_list = np.array(corners_list)
print(corners_list)

projected_points, _ = cv2.projectPoints( corners_list, Relative_R_matix, Relative_T_matix, K, dist_coeff)

for i in projected_points:
    # print(int(i[0][0]),int(i[0][1]))
    # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=2)
    cv2.drawMarker(current_frame, (int(i[0][0]),int(i[0][1])),(0,0,255), markerType=cv2.MARKER_STAR, 
    markerSize=4, thickness=2, line_type=cv2.LINE_AA)

plt.imshow(current_frame)
plt.show()