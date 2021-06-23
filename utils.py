import numpy as np
import open3d as o3d
import random, json, torch, cv2
from scipy.spatial.transform import Rotation as rotation_lib

def compute_sift(img,list_):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = []
    for i in list_:
        kp.append(cv2.KeyPoint(float(i[0]),float(i[1]),5))

    kp, des = sift.compute(img,kp)
    return kp, des  

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


def extract_pose_from_json(pose_path):
    
    pose_json = read_json_file(pose_path)
    pose = torch.from_numpy(json_to_numpy(pose_json[0])).unsqueeze(0)

    return pose



def plotter3DOpen(boxBB, rackBB, type=1, show=True):
    geometries = []
    vertices = []
    print(len(boxBB))
    print(len(boxBB[0]))
    print(boxBB[0][0].shape)
    if type == 1:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]] )  # X Y Z    
                vertices.append(np.asarray(mesh_box.vertices))                             
                geometries.append(mesh_box)
                pcd = o3d.geometry.PointCloud() 
                pcd.points = mesh_box.vertices                             
                geometries.append(pcd)
    elif type == 2:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]])  # X Y Z 
                pcd = mesh_box.sample_points_uniformly(number_of_points = 500)   
                vertices.append(np.asarray(pcd.points))                             
                geometries.append(pcd)
    elif type == 3:
        for shelfBoxes in boxBB:
            for box in shelfBoxes:
                mesh_box = o3d.geometry.TriangleMesh.create_box(width=box[3], height=box[5], depth=box[4])
                mesh_box.paint_uniform_color([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)]) 
                mesh_box.translate([box[0], box[1], box[2]])  # X Y Z    
                pcd = o3d.geometry.PointCloud() 
                pcd.points = mesh_box.vertices                             
                geometries.append(pcd)
                vertices.append(np.asarray(pcd.points))
    if show:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
        geometries.append(mesh_frame)
        o3d.visualization.draw_geometries(geometries)

    return vertices

def projectToImage(image, vertices, K):
    pts = []
    # implot = plt.imshow(image)

    for boxPoints in vertices:
        imagePts = K @ boxPoints.T
        imagePts[0,:] = imagePts[0,:] / imagePts[2,:]
        imagePts[1,:] = imagePts[1,:] / imagePts[2,:]
        imagePts = imagePts[:2,:].T

        pts.append(imagePts)
    # print(pts)
    #     plt.scatter(imagePts[:,0], imagePts[:,1], c='r', s=10)

    # plt.show()
    return pts   

def pose_2_transformation(pose):
    '''
    Convert poses to transformation matrix, pose is a vector such that
    pose[:3] is the translation vector
    pose[3:] are the rotation quarternions
    '''
    rot_mat = rotation_lib.from_quat(pose[3:]).as_matrix()
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]])  # / 1000
    transformation_mat = np.vstack(( np.hstack((rot_mat,   translation_vector)), 
                                     np.array([0, 0, 0, 1])   ))
    return transformation_mat

def harris_corner_detector_sift(img):

    operatedImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    operatedImage = np.float32(operatedImage)
    
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    dest = cv2.dilate(dest, None)

    indices = np.where(dest > 0.001 * dest.max())

    indices_list = []
    for i in range(len(indices[0])):
        indices_list.append([indices[1][i],indices[0][i]])

    kp, des = compute_sift(img, indices_list)

    return kp, des