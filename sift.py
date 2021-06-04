import cv2
import numpy as np
import matplotlib.pyplot as plt
from recon import wrapper_func
from scipy.spatial.transform import Rotation as R
from itertools import permutations as permute

K =  np.array([[293.33334351 ,           0.  ,        240.    ],
               [  0.         , 293.33334351  ,        135.    ],
               [  0.         ,  0.           ,        1.      ]])

# T_change = np.array([0, -3.6900005575561523, 0])
# T_change = np.array([[-3.6900005575561523/2.5], [0.0], [0.0]])
# T_change = np.array([[0.0], [0.0], [0.0]])
R_change = np.array([[0.0], [0.0], [0.0]])

T_change = np.array([-4.470000557556152/2.5, 0, -0.6999977035522464/6.5])
# R_change_1 = [0, 0, 0]

# R_matrix = R.from_euler('xyz', R_change, degrees=True)
# R_matrix = np.array(R_matrix.as_matrix())
# RT_matrix = np.array([[0 for i in range(4)] for j in range(3)])
# for i in range(3):
#     for j in range(4):
#         if(j!=3):
#             RT_matrix[i,j] = R_matrix[i,j]
#         else:
#             RT_matrix[i,j] = T_change[i]
#         print(RT_matrix[i,j], end = " ")
#     print()

def get_RT_matrix(R_change,T_change):
    # print(T_change)
    R_matrix = R.from_euler('xyz', R_change, degrees=True)
    R_matrix = np.array(R_matrix.as_matrix())
    RT_matrix = np.array([[0.0 for i in range(4)] for j in range(3)])
    for i in range(3):
        for j in range(4):
            if(j!=3):
                RT_matrix[i,j] = R_matrix[i,j]
            else:
                RT_matrix[i,j] = T_change[i]
    return RT_matrix

# print(K @ get_RT_matrix(R_change, T_change) @ np.array([1.921875, -1.609375,  4.953125, 1]).T)

def display_on_image(img,points_list):

    points_list = np.array(points_list)
    plt.imshow(img)
    plt.scatter(points_list[:,0], points_list[:,1], c='r', s=10)
    plt.show()

def harris_corner_detector(img):

    operatedImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    operatedImage = np.float32(operatedImage)
    
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    dest = cv2.dilate(dest, None)

    indices = np.where(dest > 0.001 * dest.max())

    indices_list = []
    for i in range(len(indices[0])):
        indices_list.append([indices[1][i],indices[0][i]])

    return indices_list

# Get specified number of matches between both images using BF matcher(Brute force matcher)
def feature_matching(img1, img2, kp1, des1, kp2, des2, list1, list2, vertices_original, numMatches = 20, viz = False):
    
    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)[:numMatches]

    img2_points = []
    img1_points = []
    obj_points = []
    dist_coeff = np.zeros((1,5))

    for m in matches:
        img2_points.append(list2[m.trainIdx])
        obj_points.append(vertices_original[m.queryIdx])
        img1_points.append(list1[m.queryIdx])


    img2_points = np.array(img2_points,dtype=np.float32)
    obj_points = np.array(obj_points,dtype=np.float32)

    # pts_1 = []
    # for boxPoints in obj_points:
    #     boxPoints = np.append(boxPoints, [1], axis = 0)
    #     # print("boxPoints : ", boxPoints)
    #     imagePts = K @ get_RT_matrix(R_change, [0,0,0]) @ boxPoints.T
    #     imagePts[0] = int(imagePts[0] / imagePts[2])
    #     imagePts[1] = int(imagePts[1] / imagePts[2])
    #     imagePts = imagePts[:2].T
    #     # print("imagePoints : ", imagePts)
    #     pts_1.append(imagePts)

    # pts_2 = []
    # for boxPoints in obj_points:
    #     boxPoints = np.append(boxPoints, [1], axis = 0)
    #     # print("boxPoints : ", boxPoints)
    #     imagePts = K @ get_RT_matrix(R_change, T_change) @ boxPoints.T
    #     imagePts[0] = int(imagePts[0] / imagePts[2])
    #     imagePts[1] = int(imagePts[1] / imagePts[2])
    #     imagePts = imagePts[:2].T
    #     print("imagePoints : ", imagePts)
    #     pts_2.append(imagePts)

    print(img2_points.shape)
    print(obj_points.shape)
    
    
    
    length = img2_points.shape[0]
    img2_points = img2_points.reshape(length, 1, 2)
    obj_points = obj_points.reshape(length, 1, 3)

    # print(img2_points.shape)
    # print(obj_points.shape)

    # (_, rotation_vector, translation_vector, inliers) = cv2.solvePnPRansac(obj_points, img2_points, K, dist_coeff)
    (_, rotation_vector, translation_vector) = cv2.solvePnP(obj_points, img2_points, K, dist_coeff, None, None, False, cv2.SOLVEPNP)
    
    # (_, rotation_vector, translation_vector) = cv2.solvePnP(obj_points, img2_points, K, dist_coeff)
    print(rotation_vector)
    print(R_change)
    print(translation_vector)

    print(rotation_vector.shape, R_change.shape)

    projected_points, _ = cv2.projectPoints( obj_points, rotation_vector, translation_vector, K, dist_coeff)
    print(projected_points)

    projected_points_GT_2, _ = cv2.projectPoints( obj_points, R_change, T_change, K, dist_coeff)

    projected_points_GT_1, _ = cv2.projectPoints( obj_points, R_change, np.array([[0.0], [0.0], [0.0]]), K, dist_coeff)

    
    if True:
        
        # for i in projected_points_GT_1:
        #     img1 = cv2.circle(img1, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=3)
        # plt.imshow(img1)
        # plt.show()

        img_2 = img2
        for i in projected_points_GT_2:
        # #     # print(i)
            img_2 = cv2.circle(img_2, (int(i[0][0]),int(i[0][1])), radius=1, color=(0, 0, 255), thickness=3)

        for i in projected_points:
        #     # print(i)
            img_2 = cv2.circle(img_2, (int(i[0][0]),int(i[0][1])), radius=1, color=(255, 0, 255), thickness=3)
        # plt.imshow(img2)
        
        # img = cv2.drawMatches(img1, kp1, img2, kp2, matches, img2, flags=2)

        # f, axarr = plt.subplots(2,2)
        # # axarr[0,0].imshow(img1)
        # axarr[0,1].imshow(img2)
        # axarr[0,0].imshow(img_2)
        plt.imshow(img_2)
        plt.show()
        # for i in img1_points:
            # print(i)
            # img1 = cv2.circle(img2, (int(i[0]),int(i[1])), radius=1, color=(0, 255, 255), thickness=3)

        # plt.scatter(imagePts[:,0], imagePts[:,1], c='r', s=10)
        
        fig = plt.figure(figsize=(16,16))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

def compute_sift(img1,list1,img2,list2,vertices_original,thresh = 0.75):

    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = []
    kp2 = []

    for i in list1:
        kp1.append(cv2.KeyPoint(i[0],i[1],3))
    for i in list2:
        kp2.append(cv2.KeyPoint(i[0],i[1],3))

    kp1, des1 = sift.compute(img1,kp1)
    kp2, des2 = sift.compute(img2,kp2)

    feature_matching(img1, img2, kp1, des1, kp2, des2, list1, list2, vertices_original)

   
RGBimg_original = cv2.imread('./blendSample_1/blendSample/1.png')
RGBimg_original = cv2.cvtColor(RGBimg_original, cv2.COLOR_BGR2RGB)
RGBimg_slanted = cv2.imread('./blendSample_1/blendSample/3.png')
RGBimg_slanted = cv2.cvtColor(RGBimg_slanted, cv2.COLOR_BGR2RGB)

imagePoints_original, vertices_original = wrapper_func() # Make sure that the paths are the same in both files.
images_points_slanted = harris_corner_detector(RGBimg_slanted)

compute_sift(RGBimg_original, imagePoints_original, RGBimg_slanted, images_points_slanted,vertices_original, 0.75)
# display_on_image(RGBimg_original,imagePoints_original)
# display_on_image(RGBimg_slanted,images_points_slanted)

# print(R.from_euler('XYZ', [np.pi/2, 0, np.pi/2]).as_matrix())