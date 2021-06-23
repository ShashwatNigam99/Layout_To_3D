import re
import numpy as np

def get_RT_matrix_from_file(file_path):
    file = open(file_path)
    lines = file.read().replace("\n", " ")
    res = re.findall(r'\<Matrix.*?\>', lines)
    RT_M = []
    for i in range(len(res)):
        RT_Matrix = []
        matrix = res[i]
        rows = re.findall(r'\(.*?\)', matrix)
        for a in rows:
            row = a.split(",")
            row[0] = row[0][1:]
            row[-1] = row[-1][:-1]
            r = []
            for m in row:
                r.append(float(m))
            RT_Matrix.append(r)
        RT_M.append(np.array(RT_Matrix))
    return RT_M

def get_Relative_matrix(list_RT_matrix):
    i = 1;
    list_of_relative_matrix = []
    while(i < len(list_RT_matrix)):
        R_previous = list_RT_matrix[i-1]
        R_current = list_RT_matrix[i]
        relative_RT = R_previous.T.dot(R_current) # fill the formula here
        list_of_relative_matrix.append(relative_RT)
        i += 1
    return list_of_relative_matrix

RT_Matrix = get_RT_matrix_from_file("./camera_positions_and_angles.txt")
RT_Relative = get_Relative_matrix(RT_Matrix)

print(len(RT_Relative))