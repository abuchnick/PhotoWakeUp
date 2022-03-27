import json
from os import getcwd
from os.path import join
import cv2

from mask import Mask
from pose_estimation import PoseEstimator
from run_smplify_x import SmplifyX
from Image_renderer import Renderer
from inverse_warp import inverse_warp

PROJECT_ROOT = realpath(join(__file__, ".."))

# TODO if have sufficient time - make it a class and implement in __call__
def warp(warp_fn, map_img):
    map_img_projected = np.zeros_like(map_img)
    for corr in warp_fn:
        map_img_projected[corr[0][1], corr[0][0], :] = map_img[corr[1][1], corr[1][0], :]
    return map_img_projected


# TODO if have sufficient time - complete configuration definition & sys.argv path to image
if __name__ == '__main__':
    # Load Configuration
    with open(join(PROJECT_ROOT, "config.json"), 'r') as cfg:
        configuration = json.load(cfg)
    images_dir_path = join(PROJECT_ROOT, "data", configuration["imagesDirectoryName"])
    input_image_path = join(images_dir_path, configuration["inputFileName"])

    mask = Mask(img_path=input_image_path,
                save_path=images_dir_path)
    segmentation = mask.create_mask()  # here refined.mask.png is created in images_temp dir

    pose_estimator = PoseEstimator()
    pose_estimator(img_path=input_image_path)  # how do we use pose estimation outputs for smplx?

    img_name = os.path.splitext(configuration["inputFileName"])[0]
    smplifyx_object = SmplifyX()
    result = smplifyx_object()[img_name][0]

    img_size = cv2.imread(input_image_path).shape[0:2]
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









