import numpy as np
import cv2 as cv
from scipy import sparse
import scipy.sparse.linalg
import timeit

class DepthMap:

    def __init__(self, mask, depth_map_coarse, normal_map):
        self. depth_map_coarse = cv.imread(depth_map_coarse)
        self.normal_map = cv.imread(normal_map)
        self.mask = cv.imread(mask)
        self.inner_pts = []
        self.boundery_pts = []
        self.depth_map_filled = None


    def classifyPoints(self):
        gray_img = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)  # to use cv.threshold the img must be a grayscale img
        threshold_used, thresholded_image = cv.threshold(gray_img, 100, 255,
                                                         cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
        contours = np.array(contours[0]).squeeze(
            1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant

        for i in range(self.depth_map_coarse.shape[0]):
            for j in range(self.depth_map_coarse.shape[1]):
                if cv.pointPolygonTest(contours, (j, i), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv.pointPolygonTest(contours, (j, i), False) == 0:
                    self.boundery_pts.append([i, j])
                    for k in range(0, 3):
                        self.depth_map_filled[i, j, k] = self.depth_map_coarse[i, j, k]


    def warp_depth_map(self):

        self.depth_map_filled = np.zeros(self.depth_map_coarse.shape, dtype=np.float32)

        self.classifyPoints()

        h = self.depth_map_coarse.shape[0]
        w = self.depth_map_coarse.shape[1]

        normal_decode = (np.array(self.normal_map).astype(np.float32) / 127.5) - 1.0

        points = self.inner_pts + self.boundery_pts

        A = sparse.lil_array((2 * len(points) + len(self.boundery_pts), h*w))
        b = np.zeros(2 * len(points) + len(self.boundery_pts))
        inx = 0
        for i in self.inner_pts:
            A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][0]
            A[inx, i[0] * w + i[1] + 1] = normal_decode[i[0], i[1]][0]
            b[inx] = normal_decode[i[0], i[1]][0]
            inx+=1

            A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][0]
            A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][0]
            normal_decode[i[0], i[1]][1]
            inx += 1

        for i in self.boundery_pts:
            A[inx, i[0] * w + i[1]] = 1
            b[inx] = self.depth_map_coarse[i[0], i[1]][0]
            inx+=1

            if [i[0], i[1] + 1] in self.inner_pts:
                A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][0]
                A[inx, i[0] * w + i[1] + 1] = normal_decode[i[0], i[1]][0]
                b[inx] = normal_decode[i[0], i[1]][0]
                inx += 1
            if [i[0] + 1, i[1] ] in points:
                A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][0]
                A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][0]
                normal_decode[i[0], i[1]][1]
                inx += 1



        print("calculated values")

        sol = sparse.linalg.lsqr(A, b)

        print("finished matrix")
        for i in self.inner_pts:
            val = sol[0][i[0] * w + i[1]]
            for j in range(0, 3):
                self.depth_map_filled[i[0], i[1], j] = val

        print("finished depth map")

        cv.imwrite('depth map.jpeg', self.depth_map_filled)



if __name__ == "__main__":
    depth_map = DepthMap(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\smpl_depth_map.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\normal.jpeg')

    depth_map.warp_depth_map()