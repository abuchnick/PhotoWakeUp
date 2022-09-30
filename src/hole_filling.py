from inverse_warp import mean_value_coordinates
import numpy as np
import cv2
from typing import List


class HoleFilling:
    def __init__(self, depth_map):
        self.inner_points = []
        self.contours_mvcs = []
        self.inner_contours = {}
        
        self.classify_points(depth_map)

    def interpolation(self, map_array, inner_contours_values):
        for y, x, contour_num in self.inner_points:
              # here we multiply Bx2 with Bx1, where B is the number of boundary points on current contour
              # so we get Bx2, and then after summing on axis 0 
            map_array[y,x] = np.sum(np.multiply(np.array(inner_contours_values[contour_num]),np.array(self.contours_mvcs[(y,x)])))
        return map_array
    
    def contours_boundery_values(self, map):
        inner_contours_map = {}
        for contour_num, inner_contour in self.inner_contours.items():
            inner_contours_map[contour_num] = [map[point[1],point[0]] for point in inner_contour]
        return inner_contours_map


    def classify_points(self, depth_map) -> List[List[List[int]]]:
        holes = (depth_map != np.inf).astype(np.uint8)
        contours, hierarchy = cv2.findContours(holes, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        i = hierarchy[0][0][2]
        while True:
            if i == -1:
                break
            self.inner_contours[i] = np.squeeze(contours[i], axis=1)
            i = hierarchy[0][i][0]
        self.contours_mvcs = {}
        for y in range(depth_map.shape[0]):
            for x in range(depth_map.shape[1]):
                for contour_num, inner_contour in self.inner_contours.items():
                    if cv2.pointPolygonTest(inner_contour, (x, y), False) == 1:
                        self.inner_points.append([y, x, contour_num])
                        point_mvc = mean_value_coordinates(org_contours_pixels=np.array(inner_contour), inner_pixel=np.array([x, y]))
                        self.contours_mvcs[(y,x)] = point_mvc
                        break  # there can be only one contour that contains (x, y)
        return


if __name__ == '__main__':
    depth_map = np.load('depth_front.npy')
    # dmin = np.min(np.where(depth_map == np.min(depth_map), float('inf'), depth_map))
    # dmax = np.max(np.where(depth_map == np.max(depth_map), float('-inf'), depth_map))
    # cv2.imshow('depth_front', (depth_map - dmin) / (dmax-dmin))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #mask = imageio.imread('refined_mask.png')
    hole_filler = HoleFilling(depth_map=depth_map)
    map = hole_filler.contours_boundery_values(map=depth_map)
    depth_map_filled = hole_filler.interpolation(depth_map,map)
    dmin = np.min(np.where(depth_map_filled == np.min(depth_map_filled), float('inf'), depth_map_filled))
    dmax = np.max(np.where(depth_map_filled == np.max(depth_map_filled), float('-inf'), depth_map_filled))
    cv2.imshow('depth_front', (depth_map_filled - dmin) / (dmax-dmin))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    

        
            
        
    
