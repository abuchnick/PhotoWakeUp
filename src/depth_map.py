import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import lsqr
import cv2
import os
from Image_renderer import map_ranges




class DepthMap:
    def __init__(self, mask, depth_map_coarse, normal_map, depth_rescale=((0, 1), (0, 1))):
        self.depth_map_coarse = np.where(
            depth_map_coarse == float('inf'),
            0.,
            map_ranges(depth_map_coarse, depth_rescale[0], depth_rescale[1]))
        self.depth_rescale = depth_rescale
        self.normals = normal_map
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        self.mask = mask

        self.inner_pts = []
        self.boundary_pts = []
        self.thresholded_image = None
        self.b = None
        self.parameter_matrix = None

    def constructEquationsMatrix(self):
        def z(x, y):
            return y * self.depth_map_coarse.shape[1] + x

        num_equ = len(self.inner_pts) * 2 + len(self.boundary_pts) * 3
        P = ssp.lil_matrix((num_equ, self.depth_map_coarse.size))
        self.b = np.zeros(num_equ)
        i = 0
        for x, y in self.inner_pts:
            nx, ny, nz = self.normals[y, x]
            if True:
                P[i, z(x, y)] = -nz
                P[i, z(x + 1, y)] = nz
                self.b[i] = nx
                i += 1
                P[i, z(x, y)] = -nz
                P[i, z(x, y + 1)] = nz
                self.b[i] = ny
                i += 1

        for x, y in self.boundary_pts:
            nx, ny, nz = self.normals[y, x]
            P[i, z(x, y)] = 100.
            self.b[i] = 100. * self.depth_map_coarse[y, x]
            i += 1
            if self.thresholded_image[y, x + 1] > 0:
                P[i, z(x, y)] = -nz
                P[i, z(x + 1, y)] = nz
                self.b[i] = nx
                i += 1
            if self.thresholded_image[y + 1, x] > 0:
                P[i, z(x, y)] = -nz
                P[i, z(x, y + 1)] = nz
                self.b[i] = ny
                i += 1
        self.parameter_matrix = P

    def classifyPoints(self):
        _, self.thresholded_image = cv2.threshold(self.mask, 100, 255, cv2.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, _ = cv2.findContours(self.thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
        contours = np.array(contours[0]).squeeze(1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant

        for i in range(self.depth_map_coarse.shape[1]):
            for j in range(self.depth_map_coarse.shape[0]):
                if cv2.pointPolygonTest(contours, (i, j), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv2.pointPolygonTest(contours, (i, j), False) == 0:
                    self.boundary_pts.append([i, j])

    def solve_depth(self):
        self.classifyPoints()
        self.constructEquationsMatrix()
        depth = lsqr(self.parameter_matrix.tocsr(), self.b,
                     x0=np.ravel(self.depth_map_coarse),
                     show=True, iter_lim=10000)[0]
        return map_ranges(depth.reshape(self.depth_map_coarse.shape), self.depth_rescale[1], self.depth_rescale[0])


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    mask = cv2.imread(os.path.join(root, 'mask.jpg'))
    normalsz = np.load(os.path.join(root, 'normals.npz'))
    normals, rescale = normalsz['normals'], normalsz['rescale']
    depth = cv2.imread(os.path.join(root, 'depth.tiff'), cv2.IMREAD_ANYDEPTH)
    depth_map_solver = DepthMap(
        mask=mask,
        depth_map_coarse=depth,
        normal_map=normals,
        depth_rescale=rescale
    )

    new_depth = np.array(depth_map_solver.solve_depth())
    # cv2.imwrite('d.tiff', new_depth)
    dmin = np.min(np.where(new_depth == np.min(
        new_depth), float('inf'), new_depth))
    dmax = np.max(np.where(new_depth == np.max(
        new_depth), float('-inf'), new_depth))
    print(dmin, dmax)
    cv2.imshow('depth', 1. - (new_depth - dmin) / (dmax-dmin))  # 1. - (new_depth - dmin) / (dmax-dmin))
    cv2.waitKey(0)
