from inverse_warp import mean_value_coordinates
import numpy as np
import cv2
from typing import List


class HoleFilling:
    def __init__(self, depth_map):
        self.inner_points = []
        self.contours_mvcs = {}
        self.inner_contours = []

        self.classify_points(depth_map)

    def __call__(self, map):
        
        inner_contours_map = []
        for inner_contour in self.inner_contours:
            inner_contours_map.append([map[point[1],point[0]] for point in inner_contour])
        
        for y, x, contour_num in self.inner_points:
              # here we multiply Bx2 with Bx1, where B is the number of boundary points on current contour
              # so we get Bx2, and then after summing on axis 0 
            boundery = np.array(inner_contours_map[contour_num])
            if boundery.ndim == 1:
                boundery = boundery.reshape(-1,1)
            mvc = np.array(self.contours_mvcs[(y,x)]).reshape(-1,1)
            map[y,x] = np.sum(np.multiply(boundery,mvc),axis=0)
        return map

    def classify_points(self, depth_map) -> List[List[List[int]]]:
        holes = (depth_map != np.inf).astype(np.uint8)
        contours, hierarchy = cv2.findContours(holes, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        i = hierarchy[0][0][2]
        while True:
            if i == -1:
                break
            self.inner_contours.append(np.squeeze(contours[i], axis=1))
            i = hierarchy[0][i][0]
        for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                for contour_num, inner_contour in enumerate(self.inner_contours):
                    if cv2.pointPolygonTest(inner_contour, (x, y), False) == 1:
                        self.inner_points.append([y, x, contour_num])
                        point_mvc = mean_value_coordinates(org_contours_pixels=np.array(inner_contour), inner_pixel=np.array([x, y]))
                        self.contours_mvcs[(y, x)] = point_mvc
                        break  # there can be only one contour that contains (x, y)


if __name__ == '__main__':
    depth_front = np.load('depth_front.npy')
    depth_back = np.load('depth_back.npy')
    skinning_image = np.load('skinning_map_image.npy')
    hole_filler = HoleFilling(depth_map=depth_front)

    depth_front_filled = hole_filler(map=depth_front)
    depth_back_filled = hole_filler(map=depth_back)
    skinning_image_filled = hole_filler(map=skinning_image)

    np.save('depth_front_filled.npy', depth_front_filled)
    np.save('depth_back_filled.npy', depth_back_filled)
    np.save('skinning_image_filled.npy', skinning_image_filled)

    dmin = np.min(np.where(depth_front_filled == np.min(depth_front_filled), float('inf'), depth_front_filled))
    dmax = np.max(np.where(depth_front_filled == np.max(depth_front_filled), float('-inf'), depth_front_filled))
    cv2.imshow('depth_front', 1. - (depth_front_filled - dmin) / (dmax-dmin))

    dmin = np.min(np.where(depth_back_filled == np.min(depth_back_filled), float('inf'), depth_back_filled))
    dmax = np.max(np.where(depth_back_filled == np.max(depth_back_filled), float('-inf'), depth_back_filled))
    cv2.imshow('depth_back', 1. - (depth_back_filled - dmin) / (dmax-dmin))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for i in range(22):
        cv2.imshow('skinning_map', skinning_image_filled[:, :, i])
        if cv2.waitKey(200) != -1:
            break
    cv2.destroyAllWindows()
