import numpy as np
import cv2 as cv
from scipy import sparse
import scipy.sparse.linalg
import timeit
from inverse_warp import get_contours


class DepthMap:

    def __init__(self, mask, depth_map_coarse, normal_map):
        self.depth_map_coarse = cv.imread(depth_map_coarse)
        self.normal_map = cv.imread(normal_map)
        self.mask = cv.imread(mask)
        self.inner_pts = []
        self.boundary_pts = []
        self.depth_map_filled = None

    def classify_points(self):
        get_contours(self.mask)
        for i in range(self.depth_map_coarse.shape[0]):
            for j in range(self.depth_map_coarse.shape[1]):
                if cv.pointPolygonTest(contours, (j, i), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv.pointPolygonTest(contours, (j, i), False) == 0:
                    self.boundary_pts.append([i, j])
                    for k in range(0, 3):
                        self.depth_map_filled[i, j, k] = self.depth_map_coarse[i, j, k]

    def warp_depth_map(self):

        self.depth_map_filled = np.zeros(self.depth_map_coarse.shape, dtype=np.float32)

        self.classify_points()

        h = self.depth_map_coarse.shape[0]
        w = self.depth_map_coarse.shape[1]

        normal_decode = (np.array(self.normal_map).astype(np.float32) / 127.5) - 1.0

        points = self.inner_pts + self.boundary_pts

        A = sparse.lil_array((4 * len(points) + len(self.boundary_pts), h * w))
        b = np.zeros(4 * len(points) + len(self.boundary_pts))
        inx = 0
        for i in self.inner_pts:
            if normal_decode[i[0], i[1]][2] < -0.15 or normal_decode[i[0], i[1]][2] > 0.15:
                A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                A[inx, i[0] * w + i[1] + 1] = normal_decode[i[0], i[1]][2]
                b[inx] = normal_decode[i[0], i[1]][0]
                inx += 1

                A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][2]
                b[inx] = normal_decode[i[0], i[1]][1]
                inx += 1

                A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                A[inx, i[0] * w + i[1] - 1] = normal_decode[i[0], i[1]][2]
                b[inx] = normal_decode[i[0], i[1]][0]
                inx += 1

                A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                A[inx, (i[0] - 1) * w + i[1]] = normal_decode[i[0], i[1]][2]
                b[inx] = normal_decode[i[0], i[1]][1]
                inx += 1
            else:
                A[inx, i[0] * w + i[1]] = normal_decode[i[0], i[1]][1] - normal_decode[i[0], i[1]][0]
                A[inx, i[0] * w + i[1] + 1] = -1 * normal_decode[i[0], i[1]][1]
                A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][0]
                b[inx] = 0
                inx += 1
                A[inx, i[0] * w + i[1]] = normal_decode[i[0], i[1]][1] - normal_decode[i[0], i[1]][0]
                A[inx, i[0] * w + i[1] - 1] = -1 * normal_decode[i[0], i[1]][1]
                A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][0]
                b[inx] = 0
                inx += 1

        for i in self.boundary_pts:
            A[inx, i[0] * w + i[1]] = 1
            b[inx] = self.depth_map_coarse[i[0], i[1]][0]
            inx += 1
            if normal_decode[i[0], i[1]][2] < -0.05 or normal_decode[i[0], i[1]][2] > 0.05:
                if [i[0], i[1] + 1] in points:
                    A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                    A[inx, i[0] * w + i[1] + 1] = normal_decode[i[0], i[1]][2]
                    b[inx] = normal_decode[i[0], i[1]][0]
                    inx += 1
                if [i[0] + 1, i[1]] in points:
                    A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                    A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][2]
                    b[inx] = normal_decode[i[0], i[1]][1]
                    inx += 1
                if [i[0], i[1] - 1] in points:
                    A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                    A[inx, i[0] * w + i[1] - 1] = normal_decode[i[0], i[1]][2]
                    b[inx] = normal_decode[i[0], i[1]][0]
                    inx += 1
                if [i[0] - 1, i[1]] in points:
                    A[inx, i[0] * w + i[1]] = -1 * normal_decode[i[0], i[1]][2]
                    A[inx, (i[0] - 1) * w + i[1]] = normal_decode[i[0], i[1]][2]
                    b[inx] = normal_decode[i[0], i[1]][1]
                    inx += 1
            else:
                if [i[0], i[1] + 1] in points and [i[0] + 1, i[1]] in points:
                    A[inx, i[0] * w + i[1]] = normal_decode[i[0], i[1]][1] - normal_decode[i[0], i[1]][0]
                    A[inx, i[0] * w + i[1] + 1] = -1 * normal_decode[i[0], i[1]][1]
                    A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][0]
                    b[inx] = 0
                    inx += 1
                if [i[0], i[1] - 1] in points and [i[0] - 1, i[1]] in points:
                    A[inx, i[0] * w + i[1]] = normal_decode[i[0], i[1]][1] - normal_decode[i[0], i[1]][0]
                    A[inx, i[0] * w + i[1] - 1] = -1 * normal_decode[i[0], i[1]][1]
                    A[inx, (i[0] + 1) * w + i[1]] = normal_decode[i[0], i[1]][0]
                    b[inx] = 0
                    inx += 1

        print("calculated values")

        sol = sparse.linalg.lsqr(A, b)

        print("finished matrix")
        for i in self.inner_pts:
            val = sol[0][i[0] * w + i[1]]
            for j in range(0, 3):
                self.depth_map_filled[i[0], i[1], j] = val

        print("finished depth map")

        cv.imwrite('depth_map.jpeg', self.depth_map_filled)


if __name__ == "__main__":
    depth_map = DepthMap(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
                         r'C:\Users\talha\Desktop\study\semester 7\depth.png',
                         r'C:\Users\talha\Desktop\study\semester 7\normals.png')

    depth_map.warp_depth_map()
