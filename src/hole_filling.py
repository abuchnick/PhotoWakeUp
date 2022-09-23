from inverse_warp import mean_value_coordinates
import numpy as np
import cv2
from typing import List
from collections import defaultdict
import imageio
import torch

class HoleFilling:
    def __init__(self, depth_map: np.ndarray):
        self.depth_map = depth_map
        self.inner_points = []
        self.contours_mvcs = []
        self.inner_contours_depth = {}
        self.inner_contours = self.classify_points()

    def __call__(self, map_array):
        for y, x, contour_num in self.inner_points:
              # here we multiply Bx2 with Bx1, where B is the number of boundary points on current contour
              # so we get Bx2, and then after summing on axis 0 
            map_array[y,x] = torch.sum(torch.mul(torch.tensor(self.inner_contours_depth[contour_num]),torch.tensor(self.contours_mvcs[(y,x)]) )).item()
        return map_array

    def classify_points(self) -> List[List[List[int]]]:

        # thresholded_mask = cv2.threshold(self.mask, 100, 1, cv2.THRESH_BINARY)[1].astype(np.uint8)
        # holes = (self.depth_map == np.inf) & thresholded_mask
        holes = (self.depth_map != np.inf).astype(np.uint8)
        contours, hierarchy = cv2.findContours(holes, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        i = hierarchy[0][0][2]
        inner_contours = {}
        while True:
            if i == -1:
                break
            inner_contours[i] = np.squeeze(contours[i], axis=1)
            self.inner_contours_depth[i] = [self.depth_map[point[1],point[0]] for point in inner_contours[i]]
            i = hierarchy[0][i][0]
        self.contours_mvcs = {}
        for y in range(self.depth_map.shape[0]):
            for x in range(self.depth_map.shape[1]):
                for contour_num, inner_contour in inner_contours.items():
                    if cv2.pointPolygonTest(inner_contour, (x, y), False) == 1:
                        self.inner_points.append([y, x, contour_num])
                        point_mvc = mean_value_coordinates(org_contours_pixels=np.array(inner_contour), inner_pixel=np.array([x, y]))
                        self.contours_mvcs[(y,x)] = point_mvc
                        break  # there can be only one contour that contains (x, y)
        return inner_contours

if __name__ == '__main__':
    depth_map = np.load('depth_front.npy')
    #mask = imageio.imread('refined_mask.png')
    hole_filler = HoleFilling(depth_map=depth_map)
    depth_map_filled = hole_filler(depth_map)
    cv2.imwrite("depth_map_filled.jpg", depth_map_filled)
    cv2.imshow('depth_map_filled.jpg','depth_map_filled.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    

        
            
        
    
