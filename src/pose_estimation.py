# %%
import os
import sys
import cv2

OPENPOSE_DIR_PATH = os.path.realpath("./lib/openpose")

try:
    sys.path.append(os.path.join(OPENPOSE_DIR_PATH,
                                 'build/python/openpose/Release'))

    os.environ['PATH'] = os.environ['PATH'] + \
        ';' + os.path.join(OPENPOSE_DIR_PATH, 'build/bin') + ';' + \
        os.path.join(OPENPOSE_DIR_PATH, 'build/x64/Release')
    import pyopenpose as op
except ImportError:
    print("Couldn't load the OpenPose library")
    raise


class PoseEstimator:
    def __init__(self, params_ovrride=None):
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict(
            model_folder="./models",
            face=True,
            hand=True,
            number_people_max=1,
            write_json="../../data/keypoints",
            write_images="../../data/openpose_images"
        )
        if params_ovrride is not None:
            self.params.update(params_ovrride)

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
            return dict(
                keypoints=datum.keypoints
            )
        finally:
            op_wrapper.stop()
            os.chdir(cwd)


if __name__ == '__main__':
    pose_estimator = PoseEstimator()
    pose_estimator(img_path="../../data/images/man.jpg")
