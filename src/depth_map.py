import numpy as np
import scipy.sparse as ssp
import cv2
import os


class DepthMap:

    def __init__(self, mask, depth_map_coarse, normal_map):
        self.depth_map_coarse = depth_map_coarse
        # decoding normals from image
        self.normals = (normal_map.astype(np.float32) / 127.5) - 1.0
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        self.mask = mask

        self.inner_pts = []
        self.boundary_pts = []
        self.outer_pts = []
        # self.filled_pts = []
        # self.depth_map_filled = np.zeros_like(depth_map_coarse)
        self.parameter_matrix = None

    def constructEquationsMatrix(self):
        def index_depth(x, y): return x*self.depth_map_coarse.shape[0] + y
        def index_normals(
            x, y, z): return x*(self.normals.shape[0]*self.normals.shape[1]) + y*self.normals.shape[1] + z
        self.parameter_matrix = ssp.lil_array(
            (self.normals.size, self.depth_map_coarse.size))
        for x, y in self.inner_pts:
            self.parameter_matrix[index_normals(
                x, y, 0), index_depth(x, y)] = -self.normals[x, y, 2]
            self.parameter_matrix[index_normals(
                x, y, 0), index_depth(x+1, y)] = self.normals[x, y, 2]
            self.parameter_matrix[index_normals(
                x, y, 1), index_depth(x, y)] = -self.normals[x, y, 2]
            self.parameter_matrix[index_normals(
                x, y, 1), index_depth(x, y+1)] = self.normals[x, y, 2]
            self.parameter_matrix[index_normals(
                x-1, y, 0), index_depth(x-1, y)] = -self.normals[x-1, y, 2]
            self.parameter_matrix[index_normals(
                x-1, y, 0), index_depth(x, y)] = self.normals[x-1, y, 2]
            self.parameter_matrix[index_normals(
                x, y-1, 1), index_depth(x, y-1)] = -self.normals[x, y-1, 2]
            self.parameter_matrix[index_normals(
                x, y-1, 1), index_depth(x, y)] = self.normals[x, y-1, 2]

        # for x, y in self.boundary_pts:
        # set boundry conditions to the coarse depth map

    def classifyPoints(self):
        _, thresholded_image = cv2.threshold(self.mask, 100, 255,
                                             cv2.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
        contours = np.array(contours[0]).squeeze(
            1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant

        for i in range(self.depth_map_coarse.shape[0]):
            for j in range(self.depth_map_coarse.shape[1]):
                if cv2.pointPolygonTest(contours, (i, j), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv2.pointPolygonTest(contours, (i, j), False) == 0:
                    self.depth_map_filled[i, j] = self.depth_map_coarse[i, j]
                    self.boundary_pts.append([i, j])
        self.normals[]

    # def solve_depth(self, x, y):

    #     n_x, n_y, n_z = self.normal_decode(self.normal_map, x, y)

    #     if (x-1, y) in self.filled_pts:
    #         # the three channels are the same
    #         b1 = self.depth_map_coarse[x-1, y, 0]
    #     else:
    #         b1 = self.depth_map_coarse[x + 1, y, 0]

    #     if (x, y-1) in self.filled_pts:
    #         b2 = self.depth_map_coarse[x, y - 1, 0]
    #     else:
    #         b2 = self.depth_map_coarse[x, y + 1, 0]

    #     A = np.array([[n_z], [n_z]])
    #     # print(np.array(A))
    #     b1 = n_z * b1 - n_x
    #     b2 = n_z * b2 - n_y
    #     b = np.array([b1, b2])
    #     # print(b)

    #     depth = np.linalg.lstsq(A, b)
    #     depth = int(depth[0])

    #     return depth

    # def warp_depth_map(self):

    #     self.classifyPoints()

    #     for i in range(len(self.boundary_pts)):
    #         x = self.boundary_pts[i][0]
    #         y = self.boundary_pts[i][1]
    #         self.filled_pts.append((x, y))
    #         for j in range(0, 3):
    #             self.depth_map_filled[x, y, j] = self.depth_map_coarse[x, y, j]

    #     #cv2.imshow('depth map', self.depth_map_filled)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     max_itr = 100

    #     height = self.depth_map_coarse.shape[0]
    #     width = self.depth_map_coarse.shape[1]

    #     for i in range(max_itr):
    #         add_count = 0

    #         # fine new points
    #         down_pt = set([((i[0] + 1), i[1])
    #                        for i in self.filled_pts if (i[0] + 1) < height]) - set(self.filled_pts)
    #         up_pt = set([((i[0] - 1), i[1])
    #                      for i in self.filled_ptsif(i[0] - 1) > -1]) - set(self.filled_pts)
    #         right_pt = set([(i[0], (i[1] - 1))
    #                         for i in self.filled_pts if (i[1] + 1) < width]) - set(self.filled_pts)
    #         left_pt = set([(i[0], (i[1] + 1))
    #                        for i in self.filled_pts if (i[1] - 1) > -1]) - set(self.filled_pts)
    #         inner_pt = set([(i[0], (i[1])) for i in self.inner_pts])

    #         up_left = up_pt.intersection(left_pt)
    #         up_right = up_pt.intersection(right_pt)
    #         up = up_left.union(up_right)

    #         down_left = down_pt.intersection(left_pt)
    #         down_right = down_pt.intersection(right_pt)
    #         down = down_left.union(down_right)

    #         new_pt = up.union(down)

    #         new_pt = list(new_pt.intersection(inner_pt))

    #         print(len(new_pt))

    #         for i in range(len(new_pt)):

    #             x = new_pt[i][0]
    #             y = new_pt[i][1]

    #             if (((x, y + 1) in self.filled_pts) or ((x, y-1) in self.filled_pts)) and (((x+1, x) in self.filled_pts) or ((x-1, y) in self.filled_pts)):
    #                 depth = self.solve_depth(x, y)
    #                 self.filled_pts.append((x, y))
    #                 add_count += 1
    #                 for j in range(0, 3):
    #                     self.depth_map_filled[x, y, j] = depth

    #             else:
    #                 continue

    #         print('fill depth map, add %d' % (add_count))

    #     cv2.imshow('depth map', self.depth_map_filled)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def solve_depth(self):
        self.classifyPoints()
        self.constructEquationsMatrix()
        depth = ssp.linalg.lstsq(self.parameter_matrix.tocsr(), np.ravel(
            self.normals), overwrite_a=True)[0]
        return depth.reshape(self.depth_map_coarse.shape)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    mask = cv2.imread(os.path.join(root, 'mask.jpg'))
    normals = cv2.imread(os.path.join(root, 'normals.jpg'))
    depth = cv2.imread(os.path.join(root, 'depth.tiff'), cv2.ANY_DEPTH)
    depth_map = DepthMap(
        mask=mask,
        depth_map_coarse=depth,
        normal_map=normals
    )

    depth_map.warp_depth_map()
