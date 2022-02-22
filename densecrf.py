import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine


image = cv2.imread('steph.png')
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device="cpu", model_folder="C:/Users/talha/Desktop/study/semester 7/פרויקט/AVR-Project") # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False, L=900)

# this line to save output
cv2.imwrite('output.png', output)

plt.imshow(output)
plt.show()