import os
import bpy
import json
import bpy_extras
from mathutils import Matrix
from mathutils import Vector

if __name__ == "__main__":

    sce = bpy.context.scene
    ob = bpy.context.object
    save_directory = "../blendSample/"

    for f in range(sce.frame_start, sce.frame_end):
        sce.frame_set(f)
        print("Frame %i" % f)
        cam_obj = bpy.data.objects['camera']
        bpy.data.objects['camera'].rotation_mode = 'QUATERNION'
        bpy.context.scene.render.filepath = save_directory+str(f).zfill(6)+".png"
        bpy.ops.render.render(write_still = True)
        camera_quaternion = cam_obj.rotation_quaternion
        camera_position = cam_obj.location
        camera_json = {"position":{"x": camera_position.x ,"y":camera_position.y, "z":camera_position.z},
                       "rotation":{"x":camera_quaternion.x,"y":camera_quaternion.y,"z":camera_quaternion.z,"w":camera_quaternion.w},
                       "frame_number" : f}
        json_file_name = "frame_%08d_CameraPose.json" % f
        json_file_name = os.path.join(save_directory, json_file_name)
        print(json_file_name)
        with open(json_file_name, "w") as fp:
            json.dump(camera_json, fp)
        
    print("Done with data generation")
