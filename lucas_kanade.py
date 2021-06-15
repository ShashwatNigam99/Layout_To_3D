import numpy as np
import cv2
from recon import wrapper_func, K
dist_coeff = np.zeros((1,5))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(200,3))

# Take first frame and find corners in it
old_frame = cv2.imread('./blendSample_video/blendSample/%s.png'%(str(1).zfill(6)))
kp1, des1, vertices_original, imagePoints_original = wrapper_func() # Make sure that the paths are the same in both files.
vertices_original = np.array(vertices_original)
imagePoints_original = np.array(imagePoints_original,dtype=np.float32)

imagePoints_original = np.reshape(imagePoints_original,(imagePoints_original.shape[0],1, imagePoints_original.shape[1]))
vertices_original = np.reshape(vertices_original,(vertices_original.shape[0],1, vertices_original.shape[1]))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0_ = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = imagePoints_original

mask = np.zeros_like(old_frame)

for i in range(2,25):

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
    print("before refining")
    print(translation_vector)
    print(rotation_matrix)


    rvec, tvec = cv2.solvePnPRefineLM(good_new_vertices, good_new, K, dist_coeff,\
         rotation_vector, translation_vector, criteria =\
         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1))
    rvec_matrix, _ = cv2.Rodrigues(rvec)
    print("after refining")
    print(tvec)
    print(rvec_matrix)
    # print(good_new_vertices[0:2])
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