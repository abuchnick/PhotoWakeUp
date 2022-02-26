# %%
import os
import sys
import cv2

OPENPOSE_DIR_PATH = os.path.realpath("../openpose/build")
try:
    sys.path.append(os.path.join(OPENPOSE_DIR_PATH,
                                 'python\\openpose\\Release'))
    os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.join(
        OPENPOSE_DIR_PATH, 'x64\\Release') + ';' + os.path.join(OPENPOSE_DIR_PATH, 'bin')
    import pyopenpose as op
except ImportError:
    print("Couldn't load the OpenPose library")
    # raise

image_path = "../data/images/LeBron_James.png"

# %%

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../openpose/models"
params["face"] = True
params["hand"] = True
params["write_json"] = "../data/keypoints"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
imageToProcess = cv2.imread(image_path)
datum.cvInputData = imageToProcess
datum.name = ".".join(os.path.basename(image_path).split(".")[0:-1])
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# %%
