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
import pickle
from animation_viewer import AnimationWindow
from camera import Camera
import moderngl_window as mglw
from hole_filling import HoleFilling


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


# Pose Estimation
pose_estimator = PoseEstimator()
file_name = ".".join(os.path.basename(input_image_path).split(".")[0:-1])
torch.cuda.empty_cache()

pose_estimator(img=input_image,
               name=file_name)  # how do we use pose estimation outputs for smplx?


# SMPLX Model Creation
smplifyx_object = SmplifyX()
img_name = os.path.splitext(configuration["inputFileName"])[0]

result = smplifyx_object()[img_name][0]
with open('result.pkl', 'wb') as file:
    pickle.dump(result, file)

# SMPLX Model: Normals, Depth and Skinning Maps rendering, and Segmentation Mask and Projection Matrix retreival
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
smpl_skinning_map = renderer.render_skinning_map(result['mesh']['skinning_map'])
smpl_projection_matrix = renderer.projection_matrix

# SMPLX Model: Hole Filling
hole_filler = HoleFilling(depth_map=smpl_depth_front)

smpl_depth_front_filled = hole_filler(map=smpl_depth_front)
smpl_depth_back_filled = hole_filler(map=smpl_depth_back)
smpl_skinning_image_filled = hole_filler(map=smpl_skinning_map)

# dmin_front = np.min(smpl_depth_front)
# dmin_back = np.min(smpl_depth_back)
# dmax_front = np.max(np.where(smpl_depth_front == np.max(smpl_depth_front), float('-inf'), smpl_depth_front))
# dmax_back = np.max(np.where(smpl_depth_back == np.max(smpl_depth_back), float('-inf'), smpl_depth_back))
# cv2.imshow('normal_front.jpg', (np.flip(smpl_normals_front, axis=2) + 1.0) / 2)
# cv2.imshow('normal_back.jpg', (np.flip(smpl_normals_back, axis=2) + 1.0) / 2)
# cv2.imshow('depth_front.jpg', 1. - (smpl_depth_front - dmin_front) / (dmax_front - dmin_front))
# cv2.imshow('depth_back.jpg', 1. - (smpl_depth_back - dmin_back) / (dmax_back - dmin_back))

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imwrite("smpl_depth_front.tiff", smpl_depth_front)
# cv2.imwrite("smpl_back_depth.tiff", smpl_depth_back)
# cv2.imwrite("smpl_mask.jpg", smpl_mask)
# np.save("smpl_normals_front.npy", smpl_normals_front)
# np.save("smpl_normals_back.npy", smpl_normals_back)


# Inverse Warp
warp_func = inverse_warp(refined_mask_img=segmentation, smpl_mask_img=smpl_mask)
# warp_func = np.load('warp.npy')
warp = Warp(warp_func)

projected_depth_front = warp(map_img=smpl_depth_front_filled)
projected_depth_back = warp(map_img=smpl_depth_back_filled)
projected_skinning_map = warp(map_img=smpl_skinning_image_filled)

# Skinning map fixes
problematic_pixels = ~((np.abs(projected_skinning_map.sum(axis=-1) - 1.0) < 0.01) | np.any(smpl_mask < 125, axis=2))
for x, y in zip(*np.nonzero(problematic_pixels)):
    value = np.zeros(projected_skinning_map.shape[-1])
    for i in (x-1, x, x+1):
        for j in (y-1, y, y+1):
            value += projected_skinning_map[i, j]
    value = value / np.sum(value)
    projected_skinning_map[x, y] = value
    
uv = renderer.get_uv_coords()
scaled_uv = (uv * np.array([[projected_skinning_map.shape[1], projected_skinning_map.shape[0]]])).astype(np.int32)
scaled_uv[:, 1] = projected_skinning_map.shape[0] - scaled_uv[:, 1]  # reverse y axis
skinning_weights = projected_skinning_map[scaled_uv[:, 1], scaled_uv[:, 0], :]
problematic_vertices = ~(np.abs(skinning_weights.sum(axis=-1) - 1.0) < 0.01)
for v in np.nonzero(problematic_vertices)[0]:
    y, x = scaled_uv[v]
    value = np.zeros(skinning_weights.shape[-1])
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            value += projected_skinning_map[i, j]
    value = value / np.sum(value)
    skinning_weights[v] = value

# projected_normals_front = warp(map_img=smpl_normals_front)
# projected_normals_back = warp(map_img=smpl_normals_back)


# np.save("normals_front.npy", projected_normals_front)
# np.save("normals_back.npy", projected_normals_back)

#
# dmin_front = np.min(np.where(projected_depth_front == float('-inf'), float('inf'), projected_depth_front))
# dmin_back = np.min(np.where(projected_depth_back == float('-inf'), float('inf'), projected_depth_back))
# dmax_front = np.max(np.where(projected_depth_front == np.max(projected_depth_front), float('-inf'), projected_depth_front))
# dmax_back = np.max(np.where(projected_depth_back == np.max(projected_depth_back), float('-inf'), projected_depth_back))
# cv2.imshow('normal_front.jpg',  (np.flip(projected_normals_front, axis=2) + 1.0) / 2)
# cv2.imshow('normal_back.jpg', (np.flip(projected_normals_back, axis=2) + 1.0) / 2)
# a = 1. - (projected_depth_front - dmin_front) / (dmax_front - dmin_front)
# print(f"{np.max(a)},{np.min(a)}")
# print(a)
# cv2.imshow('depth_front.jpg', 1. - (projected_depth_front - dmin_front) / (dmax_front - dmin_front))
# cv2.imshow('depth_back.jpg', 1. - (projected_depth_back - dmin_back) / (dmax_back - dmin_back))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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

#back_depth_integration = np.array(back_depth_solver.solve_depth())
#back_depth_integration = np.load('back_depth_integration.npy')
front_depth_integration = projected_depth_front
back_depth_integration = projected_depth_back

# Create mesh
segmentation[front_depth_integration == np.Inf] = 0
segmentation[front_depth_integration == np.NINF] = 0
mesh = Reconstruct(segmentation, front_depth_integration, back_depth_integration, smpl_projection_matrix)
vertices, faces, uv_coords = mesh.create_mesh()



# AnimationWindow.img = input_image
# AnimationWindow.mesh = dict(
#     vertices=vertices,
#     faces=faces
# )
# AnimationWindow.window_size = (input_image.shape[1], input_image.shape[0])
# AnimationWindow.camera = Camera(
#     camera_translation=result['camera']['translation'],
#     rotation_matrix=result['camera']['rotation'],
#     distance=np.linalg.norm(result['mesh']['vertices'].mean(axis=0) - result['camera']['translation']),
#     znear=1.0,
#     zfar=50.0,
#     focal_length=5000.0
# )
# # AnimationWindow.rotations_matrices = np.load('skip_to_walk_rot_mats.npy')
# print("mesh", vertices.mean(axis=0))
# mglw.run_window_config(AnimationWindow)
