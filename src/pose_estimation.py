# %%
import os
import sys
import cv2
from os.path import join, realpath

PROJECT_ROOT = realpath(join(__file__, ".."))
OPENPOSE_DIR_PATH = realpath(join(PROJECT_ROOT, "lib", "openpose"))

try:
    sys.path.append(join(OPENPOSE_DIR_PATH,
                         'build', 'python', 'openpose', 'Release'))

    os.environ['PATH'] = os.environ['PATH'] + \
                         ';' + join(OPENPOSE_DIR_PATH, 'build', 'bin') + ';' + \
                         join(OPENPOSE_DIR_PATH, 'build', 'x64', 'Release')
    import pyopenpose as op
except ImportError:
    print("Couldn't load the OpenPose library")
    raise


class PoseEstimator:
    def __init__(self, params_override=None):
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict(
            model_folder="./models",
            face=True,
            hand=True,
            number_people_max=1,
            write_json=join("..", "..", "data", "keypoints"),
            write_images=join("..", "..", "data", "openpose_images")
        )
        if params_override is not None:
            self.params.update(params_override)

    def __call__(self, img_path):
        cwd = os.getcwd()
        try:
            os.chdir(OPENPOSE_DIR_PATH)
            op_wrapper = op.WrapperPython()
            op_wrapper.configure(self.params)
            op_wrapper.start()

            datum = op.Datum()
            image_to_process = cv2.imread(img_path)
            datum.cvInputData = image_to_process
            datum.name = ".".join(os.path.basename(img_path).split(".")[0:-1])
            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        finally:
            op_wrapper.stop()
            os.chdir(cwd)


if __name__ == '__main__':
    pose_estimator = PoseEstimator()
    pose_estimator(img_path=join(PROJECT_ROOT, "data", "images_temp", "man.jpg"))
