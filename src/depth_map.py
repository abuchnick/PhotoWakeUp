import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import lsqr
import cv2
import os


class DepthMap:

    def __init__(self, mask, depth_map_coarse, normal_map):
        self.depth_map_coarse = np.where(
            depth_map_coarse == float('inf'), 0., depth_map_coarse)
        self.normals = (normal_map.astype(np.float32) / 127.5) - 1.0
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        self.mask = mask

        self.inner_pts = []
        self.boundary_pts = []
        self.contours = None
        self.b = None
        self.parameter_matrix = None

    def constructEquationsMatrix(self):
        def index_depth(y, x):
            return y * self.depth_map_coarse.shape[1] + x

        num_equ = len(self.inner_pts) * 2 + len(self.boundary_pts) * 3
        self.parameter_matrix = ssp.lil_array((num_equ, self.depth_map_coarse.size))
        self.b = np.zeros(num_equ)
        i = 0
        for x, y in self.inner_pts:
            if self.normals[y, x, 2] > 0.01:
                self.parameter_matrix[i, index_depth(y, x)] = -self.normals[y, x, 2]
                self.parameter_matrix[i, index_depth(y, x + 1)] = self.normals[y, x, 2]
                self.b[i] = self.normals[y, x, 0]
                i += 1
                self.parameter_matrix[i, index_depth(y, x)] = -self.normals[y, x, 2]
                self.parameter_matrix[i, index_depth(y + 1, x)] = self.normals[y, x, 2]
                self.b[i] = self.normals[y, x, 1]
                i += 1
            else:
                self.parameter_matrix[i, index_depth(y, x)] = self.normals[y, x, 0] - self.normals[y, x, 1]
                self.parameter_matrix[i, index_depth(y, x + 1)] = self.normals[y, x, 1]
                self.parameter_matrix[i, index_depth(y + 1, x)] = - self.normals[y, x, 0]
                i += 1

        for x, y in self.boundary_pts:
            self.parameter_matrix[i, index_depth(y, x)] = 1.
            self.b[i] = self.depth_map_coarse[y, x]
            i += 1
            if cv2.pointPolygonTest(self.contours, (x + 1, y), False) != -1\
                    and cv2.pointPolygonTest(self.contours, (x, y + 1), False) != -1:
                self.parameter_matrix[i, index_depth(y, x)] = self.normals[y, x, 0] - self.normals[y, x, 1]
                self.parameter_matrix[i, index_depth(y, x + 1)] = self.normals[y, x, 1]
                self.parameter_matrix[i, index_depth(y + 1, x)] = - self.normals[y, x, 0]
                i += 1
        print(f"{i=}")

    def classifyPoints(self):
        _, thresholded_image = cv2.threshold(self.mask, 100, 255, cv2.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
        self.contours = np.array(contours[0]).squeeze(1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant

        for i in range(self.depth_map_coarse.shape[1]):
            for j in range(self.depth_map_coarse.shape[0]):
                if cv2.pointPolygonTest(self.contours, (i, j), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv2.pointPolygonTest(self.contours, (i, j), False) == 0:
                    self.boundary_pts.append([i, j])

    def solve_depth(self):
        self.classifyPoints()
        self.constructEquationsMatrix()
        depth = lsqr(self.parameter_matrix.tocsr(), self.b,
                     x0=np.ravel(self.depth_map_coarse), show=True, iter_lim=800)[0]
        return depth.reshape(self.depth_map_coarse.shape)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    mask = cv2.imread(os.path.join(root, 'mask.jpg'))
    normals = cv2.imread(os.path.join(root, 'normals.jpg'))
    depth = cv2.imread(os.path.join(root, 'depth.tiff'), cv2.IMREAD_ANYDEPTH)
    depth_map_solver = DepthMap(
        mask=mask,
        depth_map_coarse=depth,
        normal_map=normals
    )

    new_depth = np.array(depth_map_solver.solve_depth())
    print(new_depth)
    # cv2.imwrite('d.tiff', new_depth)
    # dmin = np.min(np.where(new_depth == np.min(
    #     new_depth), float('inf'), new_depth))
    # dmax = np.max(np.where(new_depth == np.max(
    #     new_depth), float('-inf'), new_depth))
    # print(dmin, dmax)
    cv2.imshow('depth', new_depth)  # 1. - (new_depth - dmin) / (dmax-dmin))
    cv2.waitKey(0)
