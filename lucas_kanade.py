import numpy as np
import cv2
import matplotlib.pyplot as plt
from reconstruction import reconstruction

from utils import get_image_and_transformation, get_relative_tranformation_cv
from config import *

FRAMES = 60

images, transformations = get_image_and_transformation(DATA_PATH, FRAMES, list= True)


# Create some random colors
color = np.random.randint(0,255,(200,3))

# Take first frame and reconstruct it
kp1, des1, vertices_original, imagePoints_original = reconstruction(DATA_PATH)

vertices_original = np.array(vertices_original)
imagePoints_original = np.array(imagePoints_original,dtype=np.float32)
imagePoints_original = np.reshape(imagePoints_original,(imagePoints_original.shape[0],1, imagePoints_original.shape[1]))
vertices_original = np.reshape(vertices_original,(vertices_original.shape[0],1, vertices_original.shape[1]))

vertices_original_GT = vertices_original

old_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

p0 = imagePoints_original

mask = np.zeros_like(images[0])

T_loss_unrefined = []
T_loss_refined = []
R_loss_unrefined = []
R_loss_refined = []
x_axis = []

for i in range(2,FRAMES+1):

    x_axis.append(i)

    print("\n FRAME NUMBER %d \n"%(i))
    frame = images[i-1]
    frame_gray = cv2.cvtColor(images[i-1], cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    good_new_vertices = vertices_original[st==1]

    # Predictions
    (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(good_new_vertices, good_new,\
                            K, dist_coeff, reprojectionError = 1.0, flags = cv2.SOLVEPNP_EPNP)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    print("Ground truth")
    relative_transform = get_relative_tranformation_cv(transformations[i-2],transformations[i-1])
    Rotation = relative_transform[:3, :3]
    Translation = relative_transform[:3, -1]
    print(Rotation)
    print(Translation)

    print("Before refining")
    print(translation_vector.T)
    print(rotation_matrix)
    T_loss_unrefined.append(np.mean(abs(translation_vector.T-Translation)))
    R_loss_unrefined.append(np.mean(abs(rotation_matrix-Rotation))) 

    rvec, tvec = cv2.solvePnPRefineLM(good_new_vertices, good_new, K, dist_coeff,\
         rotation_vector, translation_vector, criteria =\
         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1))
    rvec_matrix, _ = cv2.Rodrigues(rvec)

    print("After refining")
    print(tvec.T)
    print(rvec_matrix)
    T_loss_refined.append(np.mean(abs(tvec.T - Translation)))
    R_loss_refined.append(np.mean(abs(rvec_matrix - Rotation))) 

    good_new_vertices = good_new_vertices @ rvec_matrix + tvec.T
    vertices_original_GT = vertices_original_GT @ Rotation + Translation

    projected_points_pred, _ = cv2.projectPoints(good_new_vertices, rvec_matrix, tvec, K, dist_coeff)
    
    projected_points_GT, _ = cv2.projectPoints(vertices_original_GT, Rotation, Translation, K, dist_coeff)
    
    # for i in projected_points_pred:
    #         cv2.drawMarker(frame, (int(i[0][0]),int(i[0][1])),(0,0,255), markerType=cv2.MARKER_STAR, 
    #         markerSize=4, thickness=2, line_type=cv2.LINE_AA)

    # for i in projected_points_GT:
    #         cv2.drawMarker(frame, (int(i[0][0]),int(i[0][1])),(255,255,255), markerType=cv2.MARKER_STAR, 
    #         markerSize=4, thickness=2, line_type=cv2.LINE_AA)

    # plt.imshow(frame)
    # plt.show()

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    vertices_original = good_new_vertices.reshape(-1,1,3)

cv2.destroyAllWindows()

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(25,15))

ax1.plot(x_axis, T_loss_unrefined, label = 'Unrefined T loss' , c = "blue")
ax1.plot(x_axis, T_loss_refined, label = 'Refined T loss' , c = "red")
ax1.legend(shadow=True, fancybox=True)
ax1.set_title("Mean of absolute difference between GT and PnP (Translation)")

ax2.plot(x_axis, R_loss_unrefined, label = 'Unrefined R loss' , c = "blue")
ax2.plot(x_axis, R_loss_refined, label = 'Refined R loss' , c = "red")
ax2.legend(shadow=True, fancybox=True)
ax2.set_title("Mean of absolute difference between GT and PnP (Rotation)")

plt.show()