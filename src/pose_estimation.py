# %%
import os
import sys
import cv2

OPENPOSE_DIR_PATH = os.path.abspath("./lib/openpose")
CWD = os.getcwd()
try:
    try:
        sys.path.append(os.path.join(OPENPOSE_DIR_PATH,
                                     'build/python/openpose/Release'))
        os.chdir(OPENPOSE_DIR_PATH)
        os.environ['PATH'] = os.environ['PATH'] + \
            ';' + os.path.join(OPENPOSE_DIR_PATH, 'build/bin') + ';' + \
            os.path.join(OPENPOSE_DIR_PATH, 'build/x64/Release')
        import pyopenpose as op
        # os.chdir(cwd)
    except ImportError:
        print("Couldn't load the OpenPose library")
        raise

    image_path = "../../data/images/steph.png"

    # %%

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./models"
    params["face"] = True
    params["hand"] = True
    params["write_json"] = "../../data/keypoints"
    params["write_images"] = "../../data/openpose_images"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()
    imageToProcess = cv2.imread(image_path)
    datum.cvInputData = imageToProcess
    datum.name = ".".join(os.path.basename(image_path).split(".")[0:-1])
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
finally:
    os.chdir(CWD)
# %%
