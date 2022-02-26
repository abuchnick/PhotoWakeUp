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
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import numpy as np


class Mask:

    def __init__(self, img_path, save_path, refine_model, detect_thresh=0.4, debug=True, device='cuda'):
        self.img_path = img_path
        self.save_path = save_path
        self.detect_tresh = detect_thresh
        self.debug = debug
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.refine_model = refine_model

        self.mask = None
        self.img = None

    def img_to_tensor(self):
        self.img = Image.open(self.img_path)
        self.img.load()
        return ToTensor()(self.img).unsqueeze(0)

    def save_img(self, img, img_name):
        img = img.detach()
        img = F.to_pil_image(img)
        img.save(f"{self.save_path}\{img_name}.png")

    def get_person_mask(self):
        print("***Creating mask***")
        image_tensor = self.img_to_tensor()
        if image_tensor.shape[1] == 4:
            image_tensor = image_tensor[:, 0:3, :, :]
        model = maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        with torch.no_grad():
            predictions = model(image_tensor)[0]
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
        print("***finished creating mask***")
        print("***Saving mask***")
        self.save_img(self.mask, "mask")

    def refine_mask(self):
        print("Start refining")
        image_np = cv2.imread(self.img_path)
        mask_np = cv2.imread(f"{self.save_path}\mask.png", cv2.IMREAD_GRAYSCALE)
        refiner = refine.Refiner(device=self.device, model_folder=self.refine_model)  # device can also be 'cpu'
        self.mask = refiner.refine(image_np, mask_np, fast=False, L=900)
        print("***Finished refining***")
        self.save_img(self.mask, "refine_mask")

    def create_mask(self):
        self.get_person_mask()
        self.refine_mask()
        return self.mask


if __name__ == "__main__":
    mask = Mask(
        img_path=r"C:\Users\talha\Desktop\study\semester 7\steph.png",
        save_path=r"C:\Users\talha\Desktop\study\semester 7",
        refine_model= r"C:\Users\talha\Desktop\study\semester 7\פרויקט\AVR-Project\models"
    )
    mask.create_mask()





