# %%
from torchvision.models.detection import maskrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn
from torchvision.models.segmentation import fcn_resnet50
import torch
import torchvision.datasets as dset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F

joints_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle']


def get_image_as_tensor(path):
    img = Image.open(path)
    img.load()
    return ToTensor()(img).unsqueeze(0)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        img.save(f"itzik{i}.jpg")
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def detect_person(path, confidence_threshold=0.8):
    image_tensor = get_image_as_tensor(path)
    if image_tensor.shape[1] == 4:
        image_tensor = image_tensor[:, 0:3, :, :]
    # model = maskrcnn_resnet50_fpn(pretrained=True)
    # model.eval() 
    model = fcn_resnet50(pretrained=True, progress=False)
    model = model.eval()

    normalized_batch = F.normalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    with torch.no_grad():
        output = model(normalized_batch)['out']
        # predictions = model(image_tensor)[0]
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    dog_and_boat_masks = [
        normalized_masks[img_idx, sem_class_to_idx[cls]]
        for img_idx in range(1)
        for cls in ('person', 'boat')
    ]

    show(dog_and_boat_masks)

    class_dim = 1
    boolean_dog_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])
    print(f"shape = {boolean_dog_masks.shape}, dtype = {boolean_dog_masks.dtype}")
    show([m.float() for m in boolean_dog_masks])
    # num_of_detections = len(predictions['labels'])
    # persons = [
    #     {'mask': predictions['masks'][i], 'box': predictions['boxes'][i]}
    #     for i in range(num_of_detections)
    #     if predictions['labels'][i] == 1 and predictions['scores'][i] > confidence_threshold
    # ]
    # areas = map(lambda person: (
    #     person['box'][2] - person['box'][0]) * (person['box'][3] - person['box'][0]), persons)
    # return persons[np.argmax(areas)]


# TODO: make sure that the joints detection and the segmentation detect the same person
def detect_joints(path):
    image_tensor = get_image_as_tensor(path)
    if image_tensor.shape[1] == 4:
        image_tensor = image_tensor[:, 0:3, :, :]
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return {
        'keypoints': predictions['keypoints'][np.argmax(predictions['scores'])],
        'keypoints_score': predictions['keypoints_scores'][np.argmax(predictions['scores'])]
    }


image = get_image_as_tensor(
    './images/LeBron_James.png').squeeze().detach().numpy().transpose(1, 2, 0)

fig, ax = plt.subplots()
ax.imshow(image)
detect_person('./images/LeBron_James.png')
# mask = detect_person('./images/LeBron_James.png')['mask']
# ax.imshow(mask.detach().numpy().transpose(1, 2, 0), cmap='jet', alpha=0.5)
# plt.show()


# fig, ax = plt.subplots()
# ax.imshow(image)
# joints = detect_joints('./images/LeBron_James.png')
# keypoints = joints['keypoints']
# keypoints_score = joints['keypoints_score']
# for (x, y, is_visible), kp_score in zip(keypoints, keypoints_score):
#     if kp_score > 2:
#         ax.plot(x, y, 'o', color='red' if int(is_visible) != 0 else 'blue')
# plt.show()

# chest_center = torch.mean(torch.vstack((keypoints[keypoints.index('right_shoulder')][:-1], keypoints[keypoints.index('left_shoulder')][:-1])), dim=0)
# hip_center = torch.mean(torch.vstack((keypoints[keypoints.index('right_hip')][:-1], keypoints[keypoints.index('left_hip')][:-1])), dim=0)
# limbs = [
#     [keypoints.index('left_eye'), keypoints.index('nose')],
#     [keypoints.index('left_eye'), keypoints.index('left_ear')],
#     [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
#     [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
#     [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
#     [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
#     [keypoints.index('right_hip'), keypoints.index('right_knee')],
#     [keypoints.index('right_knee'), keypoints.index('right_ankle')],
#     [keypoints.index('left_hip'), keypoints.index('left_knee')],
#     [keypoints.index('left_knee'), keypoints.index('left_ankle')],
#     [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
#     [keypoints.index('right_hip'), keypoints.index('left_hip')],
#     [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
#     [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
# ]
