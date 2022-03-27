import json
from os import getcwd
from os.path import join
import cv2

from mask import Mask
from pose_estimation import PoseEstimator
from run_smplify_x import SmplifyX
from Image_renderer import Renderer
from inverse_warp import inverse_warp
from depth_map import DepthMap
from create_mesh import Reconstruct

PROJECT_ROOT = realpath(join(__file__, ".."))


# TODO if have sufficient time - make it a class and implement in __call__
def warp(warp_fn, map_img):
    map_img_projected = np.zeros_like(map_img)
    for corr in warp_fn:
        map_img_projected[corr[0][1], corr[0][0], :] = map_img[corr[1][1], corr[1][0], :]
    return map_img_projected


# TODO move to other file
def uv_coordinates(_vertices, _faces, _img_size):
    img_height = _img_size[0]
    img_width = _img_size[1]
    normalized_x_coords = _vertices[0] / img_width
    normalized_y_coords = _vertices[1] / img_height
    _uv_coords = np.append(normalized_x_coords, normalized_y_coords)
    return _uv_coords


# TODO if have sufficient time - complete configuration definition & sys.argv path to image
if __name__ == '__main__':
    # Load Configuration
    with open(join(PROJECT_ROOT, "config.json"), 'r') as cfg:
        configuration = json.load(cfg)
    images_dir_path = join(PROJECT_ROOT, "data",
                           configuration["imagesDirectoryName"])
    input_image_path = join(images_dir_path,
                            configuration["inputFileName"])
    input_image = cv2.imread(input_image_path)

    mask = Mask(img_path=input_image_path,  #TODO fix pass here the input image instead of the path - happens because of inner img_to_tensor
                save_path=images_dir_path)
    segmentation = mask.create_mask()  # here refined.mask.png is created in images_temp dir

    pose_estimator = PoseEstimator()
    datum_name = ".".join(os.path.basename(input_image_path).split(".")[0:-1])
    pose_estimator(img_to_process=input_image,
                   datum_name=datum_name)  # how do we use pose estimation outputs for smplx?

    img_name = os.path.splitext(configuration["inputFileName"])[0]
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
    smpl_mask, smpl_depth = renderer.render_solid(get_depth_map=True)
    smpl_normals = renderer.render_normals()
    cv2.imwrite("smpl_depth.tiff", smpl_depth)
    cv2.imwrite("smpl_mask.jpg", smpl_mask)
    cv2.imwrite("smpl_normals.jpg", smpl_normals)
    projection_matrix = renderer.projection_matrix

    warp_func = inverse_warp(refined_mask_img=segmentation,
                             smpl_mask_img=smpl_mask)
    projected_normals = warp(warp_fn=warp_func,
                             map_img=smpl_normals)
    projected_depth = warp(warp_fn=warp_func,
                           map_img=smpl_depth)
    # TODO need to rebuild depth map using the projected normals, the integrative way
    # depth_map_constructor = DepthMap(mask=)

    # TODO need to create & project skinning map
    mesh_reconstructor = Reconstruct(mask=segmentation,
                                     depth_map_coarse=projected_depth)
    vertices, faces = mesh.create_mesh()  # TODO need to save to file..

    uv_coords = uv_coordinates(vertices, faces, img_size)

    joints = result['mesh']['joints']
