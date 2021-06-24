import numpy as np
import cv2
import matplotlib.pyplot as plt
from reconstruction import reconstruction

from config import *
# from blendSample_video.blendSample.extractRelativeRT import R_Relative, T_Relative

R_Relative = np.array(R_Relative)
T_Relative = np.array(T_Relative)

# Create some random colors
color = np.random.randint(0,255,(200,3))

# Take first frame and find corners in it
old_frame = cv2.imread('./blendSample_video/blendSample/%s.png'%(str(1).zfill(6)))
kp1, des1, vertices_original, imagePoints_original = reconstruction() # Make sure that the paths are the same in both files.
vertices_original = np.array(vertices_original)
imagePoints_original = np.array(imagePoints_original,dtype=np.float32)

imagePoints_original = np.reshape(imagePoints_original,(imagePoints_original.shape[0],1, imagePoints_original.shape[1]))
vertices_original = np.reshape(vertices_original,(vertices_original.shape[0],1, vertices_original.shape[1]))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0_ = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = imagePoints_original

mask = np.zeros_like(old_frame)

T_loss_unrefined = []
T_loss_refined = []
R_loss_unrefined = []
R_loss_refined = []
x_axis = []

for i in range(2,100):

    x_axis.append(i)

    print("\n FRAME NUMBER %d \n"%(i))
    frame = cv2.imread('./blendSample_video/blendSample/%s.png'%(str(i).zfill(6)))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    good_new_vertices = vertices_original[st==1]

    (_, rotation_vector, translation_vector,inliers) = cv2.solvePnPRansac(good_new_vertices, good_new,\
                            K, dist_coeff, reprojectionError = 1.0, flags = cv2.SOLVEPNP_EPNP)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    print("ground truth")
    print(T_Relative[i-2])
    print(R_Relative[i-2])
    print("before refining")
    print(translation_vector.T)
    print(rotation_matrix)
    T_loss_unrefined.append(np.mean(abs(translation_vector.T-T_Relative[i-2])))
    R_loss_unrefined.append(np.mean(abs(rotation_matrix-R_Relative[i-2]))) 

    rvec, tvec = cv2.solvePnPRefineLM(good_new_vertices, good_new, K, dist_coeff,\
         rotation_vector, translation_vector, criteria =\
         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1))
    rvec_matrix, _ = cv2.Rodrigues(rvec)

    print("after refining")
    print(tvec.T)
    print(rvec_matrix)
    T_loss_refined.append(np.mean(abs(tvec.T-T_Relative[i-2])))
    R_loss_refined.append(np.mean(abs(rvec_matrix-R_Relative[i-2]))) 

    good_new_vertices = good_new_vertices @ rvec_matrix + tvec.T

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(240) & 0xff
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