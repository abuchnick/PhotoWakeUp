import cv2


def inpaint_background(img_path, segmentation_mask_path, out_path):
    img = cv2.imread(filename=img_path)
    mask = cv2.imread(filename=segmentation_mask_path, flags=cv2.IMREAD_GRAYSCALE)
    result = cv2.inpaint(src=img, inpaintMask=mask, inpaintRadius=30, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(filename=out_path, img=result)