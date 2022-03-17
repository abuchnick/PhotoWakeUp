import numpy as np
import cv2 as cv
from os import getcwd
from os.path import join

IMAGES_DIR = join(getcwd(), "..", "images")
PATH_TO_ORG_MASK = join(IMAGES_DIR, "org_mask (Custom).jpeg")
PATH_TO_SMPL_MASK = join(IMAGES_DIR, "smpl_mask (Custom).jpeg")
PATH_TO_NORMALS_MAP = join(IMAGES_DIR, "normals_map (Custom).jpeg")


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
            index -= head_length
        else:
            index += arr_len + start
    return min_val, index


def get_contours(mask_img_path):
    img = cv.imread(mask_img_path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # to use cv.threshold the img must be a grayscale img
    threshold_used, thresholded_image = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
    contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
    contours = np.array(contours[0]).squeeze(1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant
    return contours
    # TODO for now both threshold_used and hierarchy vars aren't used.


def find_inner_pixels(mask_img_path, contour_points):
    img = cv.imread(mask_img_path)
    inner_pixels = []
    w, h, _ = img.shape

    for i in range(h):
        for j in range(w):
            if cv.pointPolygonTest(contour_points, (i, j), False) == 1:
                inner_pixels.append([i, j])
                # print(f"Debug: Contour Pixel in Height: {i} Width: {j}")
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
    # print(f"Debug: Mapping {m-1} -> {ind}")

    for i in reversed(range(m-1)):
        pred = int(dp_mat_pred[i, ind])
        correspondence_map[i] = pred
        # print(f"Debug: Mapping {i} -> {pred}")
        ind = pred

    return correspondence_map


def mean_value_coordinates(org_contours_pixels, inner_pixel):  # , org_contours_pixels_shifted):
    coord_diffs = org_contours_pixels - inner_pixel
    # coord_diffs_shifted = org_contours_pixels_shifted - inner_pixel
    coord_diffs_shifted = np.roll(coord_diffs, shift=(1, -1), axis=(1, 0))
    norm = np.linalg.norm(coord_diffs, ord=2, axis=1)
    # shifted_norm = np.linalg.norm(coord_diffs_shifted, ord=2, axis=1)
    shifted_norm = np.roll(norm, shift=-1, axis=0)
    coord_diffs_shifted[:, 1] *= -1
    sin = np.abs((coord_diffs * coord_diffs_shifted).sum(axis=1)) / (norm * shifted_norm)
    tan = sin / (1 + np.sqrt(1 - sin ** 2))
    tan_shifted = np.roll(tan, shift=1)
    mvc = (tan + tan_shifted) / norm
    return mvc / mvc.sum()


def inverse_warp():
    org_contours = get_contours(PATH_TO_ORG_MASK)  # shape == (m, 2)
    org_inner_pixels = find_inner_pixels(PATH_TO_ORG_MASK, org_contours)
    smpl_contours = get_contours(PATH_TO_SMPL_MASK)  # shape == (n, 2)
    correspondence = boundary_match(org_contours, smpl_contours)  # shape == (m, )
    # org_contours_pixels_shifted = np.roll(org_contours, shift=(1, -1), axis=(1, 0))  # this moves cyclically the two cols of coord_diffs upside-down and switches between the order of two cols
    result = []
    for inner_pixel in org_inner_pixels:
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
    # cv.imshow("ahlan itzik", img)
    # cv.imshow("ahlan itzik2", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    res = inverse_warp()
    img = cv.imread(PATH_TO_NORMALS_MAP)
    img2 = np.zeros_like(img)
    for r in res:
        img2[r[0][0], r[0][1], :] = img[r[1][0], r[1][1], :]
    cv.imshow("ahlan itzko", img2)
    cv.waitKey(0)
    print(1)
