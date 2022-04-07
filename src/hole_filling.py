from inverse_warp import Warp, mean_value_coordinates
import numpy as np
import cv2
from typing import List

class HoleFilling:
    def __init__(self, warp: Warp, normal_map: np.ndarray):
        self.warp = warp
        self.inner_points = []
        self.normal_map = normal_map
        self.contours_mvcs = []
        self.inner_contours = self.classify_points()

    def __call__(self, map_array):
        for y, x, contour_num in self.inner_points:
            map_array[x, y] = i  # todo continue here - i assignment is only for warning supressing

    def classify_points(self) -> List[List[List[int, int, int]]]:
        normal_mask = np.all(self.normal_map == 0, axis=2)  # TODO continuehere
        # gray_map = cv2.cvtColor(self.normal_map, cv2.COLOR_BGR2GRAY)  # to use cv2.threshold the map must be a grayscale map
        _, thresholded_map = cv2.threshold(normal_mask, 100, 255, cv2.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, hierarchy = cv2.findContours(thresholded_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        i = hierarchy[0][0][2]
        inner_contours = []
        while True:
            inner_contours.append(contours[i])
            i = hierarchy[0][i][0]
            if i == -1:
                break
        for y in range(self.normal_map.shape[0]):
            for x in range(self.normal_map.shape[1]):
                for contour_num, inner_contour in enumerate(inner_contours):
                    if cv2.pointPolygonTest(inner_contour, (x, y), False) == 1:
                        self.inner_points.append([y, x, contour_num])
                        point_mvc = mean_value_coordinates(org_contours_pixels=np.array(inner_contour), inner_pixel=np.array([x, y]))
                        self.contours_mvcs[contour_num].append(point_mvc)
                        break  # there can be only one contour that contains (x, y)
        return inner_contours

    # def calc_countours_mvcs(self) -> List[List[np.ndarray]]:
    #     inner_contours = self.classify_points()
    #     contours_mvcs = []
    #     for y, x in self.inner_points:
    #         for i, inner_contour in enumerate(inner_contours):
    #             if cv2.pointPolygonTest(inner_contour, (x, y), False) == 1:
    #                 contour_mvc = mean_value_coordinates(org_contours_pixels=np.array(inner_contour), inner_pixel=np.array([x, y]))
    #                 contours_mvcs[i].append(contour_mvc)
    #                 break  # there can be only one contour that contains (x, y)
    #     return contours_mvcs

    

        
            
        
    
