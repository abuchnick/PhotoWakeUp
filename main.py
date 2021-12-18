from torchvision.models.detection import maskrcnn_resnet50_fpn, keypointrcnn_resnet50_fpn
import torch
import torchvision.datasets as dset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor

joints_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle']


def get_image_as_tensor(path):
    img = Image.open(path)
    img.load()
    return ToTensor()(img).unsqueeze(0)


def detect_person(path, confidence_threshold=0.8):
    image_tensor = get_image_as_tensor(path)
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    num_of_detections = len(predictions['labels'])
    persons = [
        {'mask': predictions['masks'][i], 'box': predictions['boxes'][i]}
        for i in range(num_of_detections)
        if predictions['labels'][i] == 1 and predictions['scores'][i] > confidence_threshold
    ]
    areas = map(lambda person: (person['box'][2] - person['box'][0]) * (person['box'][3] - person['box'][0]), persons)
    return persons[np.argmax(areas)]


# TODO: make sure that the joints detection and the segmentation detect the same person
def detect_joints(path):
    image_tensor = get_image_as_tensor(path)
    model = keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return {
        'keypoints': predictions['keypoints'][np.argmax(predictions['scores'])],
        'keypoints_score': predictions['keypoints_scores'][np.argmax(predictions['scores'])]
    }


image = get_image_as_tensor('./images/lebron-james-run.jpg').squeeze().detach().numpy().transpose(1,2,0)
fig, ax = plt.subplots()
ax.imshow(image)
joints = detect_joints('./images/lebron-james-run.jpg')
keypoints = joints['keypoints']
keypoints_score = joints['keypoints_score']
for (x, y, is_visible), kp_score in zip(keypoints, keypoints_score):
    if kp_score > 2:
        ax.plot(x, y, 'o', color='red' if int(is_visible) != 0 else 'blue')
plt.show()

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
