import numpy as np
import cv2 as cv
from os import getcwd
from os.path import join

IMAGES_DIR = join(getcwd(), "..", "images")
PATH_TO_ORG_MASK = join(IMAGES_DIR, "output.png")
PATH_TO_SMPL_MASK = join(IMAGES_DIR, "smpl_mask.png")


def get_sub_array(arr, start, end):
    if start >= 0:
        sub_array = arr[start:end]
    else:
        arr_len = arr.size
        sub_array = np.append(arr[start % arr_len:], arr[0:end])
    return sub_array


def get_contours(mask_img_path):
    img = cv.imread(mask_img_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # to use cv.threshold the img must be a grayscale img
    threshold_used, thresholded_image = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
    contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
    contours = np.array(contours[0]).squeeze(1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant
    return contours
    # TODO for now both threshold_used and hierarchy vars aren't used.


def find_contours_pixels(mask_img_path, contour_points):
    img = cv.imread(mask_img_path)
    contours_pixels = []
    w, h, _ = img.shape

    for i in range(h):
        for j in range(w):
            if cv.pointPolygonTrest(contour_points, (i, j), False) == 1:
                contours_pixels.append([i, j])
                print(f"Debug: Contour Pixel in Height: {i} Width: {j}")
    return contours_pixels


def boundary_match(org_contours, smpl_contours, k=32):
    m = org_contours.shape[0]
    n = smpl_contours.shape[0]

    dp_mat_val = np.zeros((m, n))
    dp_mat_pred = np.zeros((m, n))  # save parent node of current node

    dp_mat_val += float('inf')  # set init. val to inf
    dp_mat_pred -= 1  # set parent node of all nodes to -1

    dp_mat_val[0] = np.linalg.norm(org_contours[0] - smpl_contours)
    dp_mat_pred[0] = np.arange(stop=n)

    for i in range(1, m):
        for j in range(n):
            sub_arr = get_sub_array(dp_mat_val[i - 1, :], j - k, j + 1)
            ind = np.argmin(sub_arr)
            min_val = sub_arr[ind]

            dp_mat_val[i, j] = min_val + np.linalg.norm(org_contours[i] - smpl_contours[j])
            dp_mat_pred[i, j] = ind
        print(f"Debug: Iteration {i}/{m}")

    ind = np.argmin(dp_mat_val[m-1, :])
    correspondence = dict([(m-1, ind)])
    print(f"Debug: Mapping {m-1} -> {ind}")

    for i in reversed(range(m-1)):
        pred = dp_mat_pred[i-1, ind]
        correspondence[i] = pred
        print(f"Debug: Mapping {i} -> {pred}")
        ind = pred

    return correspondence


if __name__ == '__main__':
    _org_contours = get_contours(PATH_TO_ORG_MASK)
    _smpl_contours = get_contours(PATH_TO_SMPL_MASK)
    boundary_match(_org_contours, _smpl_contours)
