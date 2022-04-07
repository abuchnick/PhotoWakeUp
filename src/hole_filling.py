from inverse_warp import mean_value_coordinates
import numpy as np
import cv2
from typing import List

class HoleFilling:
    def __init__(self, normal_map: np.ndarray):
        self.normal_map = normal_map
        self.inner_points = []
        self.contours_mvcs = []
        self.inner_contours = self.classify_points()

    def __call__(self, map_array):
        for y, x, contour_num in self.inner_points:
              # here we multiply Bx3 with Bx1, where B is the number of boundary points on current contour
              # so we get Bx3, and then after summing on axis 0  
            map_array[x, y] = (self.inner_contours[contour_num] * self.contours_mvcs[contour_num]).sum(axis=0)
        return map_array

    def classify_points(self) -> List[List[List[int]]]:
        normal_mask = np.all(self.normal_map == 0, axis=2)
        normal_mask = normal_mask.astype(np.uint8) * 255
        # _, thresholded_mask = cv2.threshold(normal_mask, 1, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(normal_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        i = hierarchy[0][0][2]
        inner_contours = []
        while True:
            if i == -1:
                break
            inner_contours.append(contours[i])
            i = hierarchy[0][i][0]
        inner_contours_num = len(inner_contours)
        self.contours_mvcs = [ [] for _ in range(inner_contours_num) ]
        for y in range(self.normal_map.shape[0]):
            for x in range(self.normal_map.shape[1]):
                for contour_num, inner_contour in enumerate(inner_contours):
                    if cv2.pointPolygonTest(inner_contour, (x, y), False) == 1:
                        self.inner_points.append([y, x, contour_num])
                        point_mvc = mean_value_coordinates(org_contours_pixels=np.array(inner_contour), inner_pixel=np.array([x, y]))
                        self.contours_mvcs[contour_num].append(point_mvc)
                        break  # there can be only one contour that contains (x, y)
        return inner_contours

if __name__ == '__main__':
    normal_map = np.load('normals_front.npy')
    hole_filler = HoleFilling(normal_map=normal_map)
    normal_map_filled = hole_filler(normal_map)



    

        
            
        
    
