import re
import numpy as np

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

def get_Relative_matrix(list_R_matrix, list_T_matrix):
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

# R_Matrix, T_Matrix = get_RT_matrix_from_file("./camera_positions_and_angles.txt")
R_Matrix, T_Matrix = get_RT_matrix_from_file("./blendSample_video/blendSample/camera_positions_and_angles.txt")
R_Relative, T_Relative = get_Relative_matrix(R_Matrix, T_Matrix)

# count = 2
# for i, j in zip(R_Relative, T_Relative):
#     print(count)
#     count+=1
#     print()
#     print(i)
#     print()
#     print(j)
#     print()
