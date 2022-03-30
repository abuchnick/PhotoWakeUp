from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
import torch
import torchvision.datasets as dset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import cv2
import time
import segmentation_refinement as refine
from os.path import join, realpath
from os import getcwd

MODELS_DIR = realpath(join(__file__, "..", "models"))


class Mask:

    def __init__(self, img_path, save_path, is_debug=True, detect_thresh=0.4, device='cuda'):
        self.img_path = img_path
        self.save_path = save_path
        self.refine_model = MODELS_DIR
        self.detect_thresh = detect_thresh
        self.is_debug = is_debug  # if set to True, save all outputs during the process
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.mask = None
        self.img = None

    def img_to_tensor(self):
        self.img = Image.open(self.img_path)
        self.img.load()
        return ToTensor()(self.img).unsqueeze(0)

    def save_img(self, img, img_name):
        img = img.detach()
        img = F.to_pil_image(img)
        img.save(join(self.save_path, img_name + ".png"))

    def get_person_mask(self):
        print("***Creating mask***")
        image_tensor = self.img_to_tensor()
        if image_tensor.shape[1] == 4:
            image_tensor = image_tensor[:, 0:3, :, :]
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        with torch.no_grad():
            predictions = model(image_tensor)[0]
        # take the mask of the person with the largest bounding box in frame
        num_of_detections = len(predictions['labels'])
        persons = [
            {'mask': predictions['masks'][i], 'box': predictions['boxes'][i]}
            for i in range(num_of_detections)
            if predictions['labels'][i] == 1 and predictions['scores'][i] > self.detect_tresh
        ]
        areas = list(map(lambda person: (
                                            torch.abs(person['box'][2] - person['box'][0])) * (
                                            torch.abs(person['box'][3] - person['box'][1])), persons))

        self.mask = (persons[np.argmax(areas)]['mask'] > 0.5).float()
        print("***Finished creating mask***")
        if self.is_debug:
            print("***Saving mask***")
            self.save_img(self.mask, "mask")

    def refine_mask(self):
        print("***Start refining***")
        image_np = cv2.imread(self.img_path)
        mask_np = cv2.imread(join(self.save_path, "mask.png"), cv2.IMREAD_GRAYSCALE)
        refiner = refine.Refiner(device=self.device, model_folder=self.refine_model)  # device can also be 'cpu'
        self.mask = refiner.refine(image_np, mask_np, fast=False, L=900)
        print("***Finished refining***")
        if self.is_debug:
            cv2.imwrite(join(self.save_path, "refined_mask.png"), self.mask)

    def create_mask(self):
        self.get_person_mask()
        self.refine_mask()
        return self.mask


if __name__ == "__main__":
    mask = Mask(
        img_path=r"C:\Users\talha\Desktop\study\semester 7\steph.png",
        save_path=r"C:\Users\talha\Desktop\study\semester 7"
    )
    mask.create_mask()





