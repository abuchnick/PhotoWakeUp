import json
from ntpath import realpath
import os
from os import getcwd
from os.path import join
import cv2
import numpy as np
import torch.cuda

from mask import Mask
from pose_estimation import PoseEstimator
from run_smplify_x import SmplifyX
from Image_renderer import Renderer
from inverse_warp import inverse_warp, Warp
from depth_map import DepthMap
from create_mesh import Reconstruct
import realpath



PROJECT_ROOT = os.path.abspath(join(__file__, "..", ".."))

# TODO if have sufficient time - complete configuration definition & sys.argv path to image

# torch.cuda.empty_cache()
# Load Configuration
with open(join(PROJECT_ROOT, "config.json"), 'r') as cfg:
    configuration = json.load(cfg)
images_dir_path = join(PROJECT_ROOT, "data",
                        configuration["imagesDirectoryName"])
images_temp_dir_path = join(PROJECT_ROOT, "data",
                        configuration["imagesTempDirectoryName"])
input_image_path = join(images_dir_path,
                        configuration["inputFileName"])
input_image = cv2.imread(input_image_path)

# Segmentation
mask = Mask(img_path=input_image_path,  # TODO fix pass here the input image instead of the path - happens because of inner img_to_tensor
            save_path=images_temp_dir_path)
segmentation = mask.create_mask()  # here refined.mask.png is created in images_temp dir


# pose estimator
pose_estimator = PoseEstimator()
file_name = ".".join(os.path.basename(input_image_path).split(".")[0:-1])
torch.cuda.empty_cache()
pose_estimator(img=input_image,
                name=file_name)  # how do we use pose estimation outputs for smplx?

img_name = os.path.splitext(configuration["inputFileName"])[0]

# SMPL

smplifyx_object = SmplifyX()
result = smplifyx_object()[img_name][0]

img_size = input_image.shape[0:2]
renderer = Renderer(
    vertices=result['mesh']['vertices'],
    faces=result['mesh']['faces'],
    img_shape=img_size,
    camera_translation=result['camera']['translation'],
    camera_rotation=result['camera']['rotation']
)
smpl_mask, smpl_depth_front = renderer.render_solid(get_depth_map=True)
_, smpl_depth_back = renderer.render_solid(get_depth_map=True, back_side=True)
smpl_normals_front, rescale_front = renderer.render_normals()
smpl_normals_back, rescale_back = renderer.render_normals(back_side=True)
skinning_map = renderer.render_skinning_map(result['mesh']['skinning_map'])

cv2.imwrite("smpl_depth_front.tiff", smpl_depth_front)
cv2.imwrite("smpl_back_depth.tiff", smpl_depth_back)
cv2.imwrite("smpl_mask.jpg", smpl_mask)
np.save("smpl_normals_front.npy", smpl_normals_front)
np.save("smpl_normals_back.npy", smpl_normals_back)

projection_matrix = renderer.projection_matrix

# Inverse warp
#warp_func = inverse_warp(refined_mask_img=segmentation, smpl_mask_img=smpl_mask)
warp_func = np.load('warp.npy')
warp = Warp(warp_func)

projected_normals_front = warp(map_img=smpl_normals_front)
projected_normals_back = warp(map_img=smpl_normals_back)

projected_depth_front = warp(map_img=smpl_depth_front)
projected_depth_back = warp(map_img=smpl_depth_back)

np.save("normals_front.npy", projected_normals_front)
np.save("normals_back.npy", projected_normals_back)

# Depth map integration

# front_depth_solver = DepthMap(
#     mask=segmentation,
#     depth_map_coarse=projected_depth_front,
#     normal_map=projected_normals_front,
#     depth_rescale=rescale_front
# )

# back_depth_solver = DepthMap(
#     mask=segmentation,
#     depth_map_coarse=projected_depth_back,
#     normal_map=projected_normals_back,
#     depth_rescale=rescale_back
# )

#front_depth_integration = np.array(front_depth_solver.solve_depth())
#front_depth_integration = np.load('front_depth_integration.npy')
front_depth_integration = projected_depth_front
back_depth_integration = projected_depth_back

#back_depth_integration = np.array(back_depth_solver.solve_depth())
#back_depth_integration = np.load('back_depth_integration.npy')

# Create mesh

mesh = Reconstruct(segmentation, front_depth_integration, back_depth_integration, projection_matrix)
vertices, faces, uv_coords = mesh.create_mesh()


joints = result['mesh']['joints']
