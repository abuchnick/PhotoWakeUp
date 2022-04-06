import cv2

img_path = r'./data/images_temp/man.jpg'
mask_path = r'./data/images_temp/man_refine_mask.jpeg'
out_path = r'./data/images_temp/man_inpainted.png'

img = cv2.imread(filename=img_path)
mask = cv2.imread(filename=mask_path, flags=cv2.IMREAD_GRAYSCALE)

result = cv2.inpaint(src=img, inpaintMask=mask, inpaintRadius=30, flags=cv2.INPAINT_TELEA)

cv2.imwrite(filename=out_path, img=result)
# cv2.imshow('inpainted', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()