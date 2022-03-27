import numpy as np
import cv2
from os import getcwd
from os.path import join
import matplotlib.pyplot as plt
from itertools import chain

IMAGES_DIR = join(getcwd(), "..", "images")
PATH_TO_ORG_MASK = join(IMAGES_DIR, "refined_mask.png")  # TODO still doesn't have this file
# PATH_TO_ORG_MASK = join(IMAGES_DIR, "org_mask (Custom).jpeg")
PATH_TO_SMPL_MASK = join(IMAGES_DIR, "smpl_mask.jpeg")
# PATH_TO_SMPL_MASK = join(IMAGES_DIR, "smpl_mask (Custom).jpeg")
PATH_TO_NORMALS_MAP = join(IMAGES_DIR, "normals_map (Custom).jpeg")
PATH_TO_DEPTH_MAP = join(IMAGES_DIR, "smpl_depth_map (Custom).jpeg")


def argmin_sub_array(arr, start, end):
    if start >= 0:
        sub_array = arr[start:end]
        index = np.argmin(sub_array)
        min_val = sub_array[index]
        index += start  # in order to get index inside the original array
    else:
        arr_len = arr.size
        sub_array = np.append(arr[start % arr_len:], arr[0:end])
        head_length = arr_len - (start % arr_len)
        index = np.argmin(sub_array)
        min_val = sub_array[index]
        if index >= head_length:
            index -= head_length  # in order to get index inside the original array
        else:
            index += arr_len + start  # in order to get index inside the original array
    return min_val, index


def get_contours(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # to use cv2.threshold the img must be a grayscale img
    _, thresholded_image = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
    contours = np.array(contours[0]).squeeze(1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant
    return contours


def find_inner_pixels(img, contour_points):
    inner_pixels = []
    h, w, _ = img.shape

    for i in range(w):
        for j in range(h):
            if cv2.pointPolygonTest(contour_points, (i, j), False) == 1:
                inner_pixels.append([i, j])
    return np.array(inner_pixels)


def boundary_match(org_contours, smpl_contours, k=32):
    m = org_contours.shape[0]
    n = smpl_contours.shape[0]

    dp_mat_val = np.zeros((m, n))
    dp_mat_pred = np.zeros((m, n), dtype=np.int32)  # save parent node of current node

    dp_mat_val += float('inf')  # set init. val to inf
    dp_mat_pred -= 1  # set parent node of all nodes to -1

    dp_mat_val[0] = np.linalg.norm(org_contours[0] - smpl_contours, ord=2, axis=1)
    dp_mat_pred[0] = np.arange(start=0, stop=n)

    for i in range(1, m):
        for j in range(n):
            min_val, index = argmin_sub_array(dp_mat_val[i - 1, :], j - k, j + 1)

            dp_mat_val[i, j] = min_val + np.linalg.norm(org_contours[i] - smpl_contours[j])
            dp_mat_pred[i, j] = index
        if i % 100 == 0:
            print(f"Debug: Iteration {i}/{m}")

    ind = np.argmin(dp_mat_val[m-1, :])
    correspondence_map = np.zeros(m, dtype=np.int32)
    correspondence_map[m-1] = ind

    for i in reversed(range(m-1)):
        pred = int(dp_mat_pred[i, ind])
        correspondence_map[i] = pred
        ind = pred

    return correspondence_map


def mean_value_coordinates(org_contours_pixels, inner_pixel):
    coord_diffs = org_contours_pixels - inner_pixel
    coord_diffs_shifted = np.roll(coord_diffs, shift=(1, -1), axis=(1, 0))  # this moves cyclically the two cols of coord_diffs upside-down and switches between the order of two cols
    norm = np.linalg.norm(coord_diffs, ord=2, axis=1)
    if np.any(norm == 0.0):
        return np.where(norm == 0, 1, 0)  # there can be only one input in norm that equals 0
    shifted_norm = np.roll(norm, shift=-1, axis=0)
    coord_diffs_shifted[:, 1] *= -1
    sin_a = (coord_diffs * coord_diffs_shifted).sum(axis=1) / (norm * shifted_norm)
    cos_a = ((norm ** 2) + (shifted_norm ** 2) - (norm-shifted_norm) ** 2) / (2 * norm * shifted_norm)
    tan_half_a = sin_a / (1 + cos_a)
    tan_shifted = np.roll(tan_half_a, shift=1)
    mvc = (tan_half_a + tan_shifted) / norm
    return mvc / mvc.sum()


def inverse_warp(refined_mask_img, smpl_mask_img):
    org_contours = get_contours(refined_mask_img)  # shape == (m, 2)
    org_inner_pixels = find_inner_pixels(refined_mask_img, org_contours)
    smpl_contours = get_contours(smpl_mask_img)  # shape == (n, 2)
    correspondence = boundary_match(org_contours, smpl_contours)  # shape == (m, )
    result = []
    for inner_pixel in chain(org_inner_pixels, org_contours):
        mvc = mean_value_coordinates(org_contours, inner_pixel)  # , org_contours_pixels_shifted)  # shape == (m, )
        transformed_pixels = np.expand_dims(mvc, axis=1) * np.take(a=smpl_contours, indices=correspondence, axis=0)  # shape == (m, 2)
        mapped_pixel = transformed_pixels.sum(axis=0)  # shape == (2, )
        result.append([inner_pixel, mapped_pixel.astype(int)])
    return result


if __name__ == '__main__':
    # TEST BOUNDARY MATCHING
    # org_contour = get_contours(PATH_TO_ORG_MASK)
    # smpl_contour = get_contours(PATH_TO_SMPL_MASK)
    # corr = boundary_match(org_contour, smpl_contour)
    # img = np.zeros((326, 500, 3))
    # img2 = np.zeros((326, 500, 3))
    # for i, j in enumerate(corr):
    #     img[org_contour[i][1], org_contour[i][0], 0] = smpl_contour[j][1] / 326
    #     img[org_contour[i][1], org_contour[i][0], 2] = smpl_contour[j][0] / 500
    #     img2[smpl_contour[j][1], smpl_contour[j][0], 0] = smpl_contour[j][1] / 326
    #     img2[smpl_contour[j][1], smpl_contour[j][0], 2] = smpl_contour[j][0] / 500
    # cv2.imshow("ahlan itzik", img)
    # cv2.imshow("ahlan itzik2", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # NORMALS INVERSE WARP
    # f = inverse_warp()
    # normals_map = cv2.imread(PATH_TO_NORMALS_MAP)
    # normals_map_projected = np.zeros_like(normals_map)
    # depth_map = cv2.imread(PATH_TO_DEPTH_MAP)
    # depth_map_projected = np.zeros_like(depth_map)
    # for corr in f:
    #     normals_map_projected[corr[0][1], corr[0][0], :] = normals_map[corr[1][1], corr[1][0], :]
    #     depth_map_projected[corr[0][1], corr[0][0], :] = depth_map[corr[1][1], corr[1][0], :]
    # cv2.imwrite(join(IMAGES_DIR, "depth_projected.png"), depth_map_projected)
    # cv2.imwrite(join(IMAGES_DIR, "normals_projected.png"), normals_map_projected)
    # cv2.imshow("ahlan normals", normals_map_projected)
    # cv2.imshow("ahlan depth", depth_map_projected)
    # cv2.waitKey(0)
    print(1)
