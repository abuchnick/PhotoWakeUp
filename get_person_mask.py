from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
import torch
import torchvision.datasets as dset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F


def get_image_as_tensor(path):
    img = Image.open(path)
    img.load()
    return ToTensor()(img).unsqueeze(0)


def save(img, img_name):
    img = img.detach()
    img = F.to_pil_image(img)
    img.save(f"{img_name}.jpg")


def get_person_mask(path, confidence_threshold=0.4):
    image_tensor = get_image_as_tensor(path)
    if image_tensor.shape[1] == 4:
        image_tensor = image_tensor[:, 0:3, :, :]
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    # model = fcn_resnet50(pretrained=True, progress=False)
    # model = model.eval()

    with torch.no_grad():
        # output = model(normalized_batch)['out']
        predictions = model(image_tensor)[0]

    # sem_classes = [
    # '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    # 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    # 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    #
    # normalized_masks = torch.nn.functional.softmax(output, dim=1)
    #
    # person_masks = [normalized_masks[0, sem_class_to_idx['person']]]
    #
    # save(person_masks, "norm_mask")
    #
    # class_dim = 1
    # boolean_person_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])
    # print(f"shape = {boolean_person_masks.shape}, dtype = {boolean_person_masks.dtype}")
    # save([m.float() for m in boolean_person_masks], "bool_mask")

    num_of_detections = len(predictions['labels'])
    persons = [
        {'mask': predictions['masks'][i], 'box': predictions['boxes'][i]}
        for i in range(num_of_detections)
        if predictions['labels'][i] == 1 and predictions['scores'][i] > confidence_threshold
    ]
    areas = list(map(lambda person: (
        torch.abs(person['box'][2] - person['box'][0])) * (torch.abs(person['box'][3] - person['box'][1])), persons))
    largest_person_mask = persons[np.argmax(areas)]['mask']
    save((largest_person_mask > 0.5).float(), "largest")


get_person_mask('./images/steph.png')


