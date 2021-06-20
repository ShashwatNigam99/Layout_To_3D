import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector

def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

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
    print(rot_mat)
    # rot_mat = np.array(mathutils.Quaternion((pose[6], pose[3], pose[4], pose[5])).to_matrix())

    print(mathutils.Quaternion((pose[6], pose[3], pose[4], pose[5])).to_matrix())
    translation_vector = np.array([[pose[0]], [pose[1]], [pose[2]]]) # / 1000


    transformation_mat = np.vstack((np.hstack((rot_mat,   translation_vector  ) ), np.array([0, 0, 0, 1])))

    return transformation_mat @ flip_y 


sce = bpy.context.scene
ob = bpy.context.object

camera_positions_and_angles_file = open("../blendSample/camera_positions_and_angles.txt", "w")
for f in range(sce.frame_start, sce.frame_end):
    sce.frame_set(f)
    print("Frame %i" % f)
    cam = bpy.data.objects['camera']
    bpy.context.scene.render.filepath = "../blendSample/"+str(f).zfill(6)+".png"
    bpy.ops.render.render(write_still = True)
    RT = get_3x4_RT_matrix_from_blender(cam)
    camera_positions_and_angles_file.write(str(f))
    camera_positions_and_angles_file.write("\n")
    camera_positions_and_angles_file.write(str(cam.location))
    camera_positions_and_angles_file.write("\n")
    camera_positions_and_angles_file.write(str(cam.rotation_euler))
    camera_positions_and_angles_file.write("\n")
    camera_positions_and_angles_file.write(str(cam.rotation_quaternion))
    camera_positions_and_angles_file.write("\n")
    camera_positions_and_angles_file.write(str(RT))
    camera_positions_and_angles_file.write("\n")

print("Here")
camera_positions_and_angles_file.close()