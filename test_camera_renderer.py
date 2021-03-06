import os
import json
import torch
import cv2
import mathutils
import numpy as np
import matplotlib.pyplot as plt
from recon import wrapper_func
FLOAT_EPS = np.finfo(np.float).eps

def read_json_file(path):
    '''
    Read json file
    '''
    json_data = []

    with open(path) as fp:
        for json_object in fp:
            json_data.append(json.loads(json_object))
    
    return json_data


def json_to_numpy(pose):
    pose = np.array([pose['position']['x'], 
                     pose['position']['y'], 
                     pose['position']['z'],
                     pose['rotation']['x'], 
                     pose['rotation']['y'], 
                     pose['rotation']['z'], 
                     pose['rotation']['w'] 
                     ])
    return pose

def extract_pose(pose_path):

    pose_json = read_json_file(pose_path)
    pose = torch.from_numpy(json_to_numpy(pose_json[0])).unsqueeze(0)

    return pose

def quat2mat(quat):
    
    x, y, z, w = quat
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def pose_2_transformation(pose):

    '''
    Convert poses to transformation matrix
    '''

  
    # pose[5] *= -1
    rot_mat_2 = np.array([[ 1,  0, 0, 0],
                          [ 0,  0, 1, 0],
                          [ 0, -1, 0, 0],
                          [ 0,  0, 0, 1]])
    
    
    flip_y = np.eye(4)
    flip_y[1,1] *= -1
    flip_y[2,2] *= -1

    flip_x = np.eye(4)
    flip_x[0, 0] *= -1
    flip_x[1, 1] *= -1

    rot_mat = quat2mat(pose[3:])
    # print(rot_mat)
    # rot_mat = np.array(mathutils.Quaternion((pose[6], pose[3], pose[4], pose[5])).to_matrix())

    # print(mathutils.Quaternion((pose[6], pose[3], pose[4], pose[5])).to_matrix())
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]]) # / 1000


    transformation_mat = np.vstack((np.hstack((rot_mat,   translation_vector  ) ), np.array([0, 0, 0, 1])))

    return transformation_mat @ flip_y 

kp1, des1, corners_list, _ = wrapper_func() # Make sure that the paths are the same in both files.
dist_coeff = np.zeros((1,5))
K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

frame_number = 1

save_directory = "../blendSample/"

json_file_name = "frame_%08d_CameraPose.json" % frame_number
json_file_name = os.path.join(save_directory, json_file_name)

pose = extract_pose(json_file_name).squeeze()
transformation = pose_2_transformation(pose)

transformation = np.array(transformation[:-1])

Rotation = transformation.T[:-1].T
Translation = transformation.T[-1]

# print(transformation)
# print(Rotation)
# print(Translation)

dist_coeff = np.zeros((1,5))
corners_list = np.array(corners_list)
# print(cube_points)
projected_points, _ = cv2.projectPoints( corners_list, Rotation, Translation, K, dist_coeff)
# print(projected_points)

current_frame = cv2.imread("../blendSample/"+str(frame_number).zfill(6)+".png")
current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

for i in projected_points:
    print(int(i[0][0]),int(i[0][1]))
    # img2 = cv2.circle(img2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=2)
    cv2.drawMarker(current_frame, (int(i[0][0]), int(i[0][1])),(0,0,255), markerType=cv2.MARKER_STAR, 
    markerSize=4, thickness=2, line_type=cv2.LINE_AA)

plt.imshow(current_frame)
plt.show()