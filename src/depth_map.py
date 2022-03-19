import numpy as np
import cv2 as cv
import time


class DepthMap:

    def __init__(self, mask, depth_map_coarse, normal_map):
        self. depth_map_coarse = cv.imread(depth_map_coarse)
        self.normal_map = cv.imread(normal_map)
        self.mask = cv.imread(mask)
        self.inner_pts = []
        self.boundery_pts = []
        self.filled_pts = []
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


    def normal_decode(self, image, x, y):

        x_axis = image[x, y, 2] * 2 - 255  # channel order of opencv imread is BGR
        y_axis = image[x, y, 1] * 2 - 255
        z_axis = image[x, y, 0] * 2 - 255

        return x_axis, y_axis, z_axis

    def solve_depth(self, x, y):

        n_x, n_y, n_z = self.normal_decode(self.normal_map, x, y)

        if (x-1, y) in self.filled_pts:
            b1 = self.depth_map_coarse[x-1, y, 0]  # the three channels are the same
        else:
            b1 = self.depth_map_coarse[x + 1, y, 0]

        if (x, y-1) in self.filled_pts:
            b2 = self.depth_map_coarse[x, y - 1, 0]
        else:
            b2 = self.depth_map_coarse[x, y + 1, 0]

        A = np.array([[n_z], [n_z]])
        # print(np.array(A))
        b1 = n_z * b1 - n_x
        b2 = n_z * b2 - n_y
        b = np.array([b1, b2])
        # print(b)

        depth = np.linalg.lstsq(A, b)
        depth = int(depth[0])

        return depth

    def warp_depth_map(self):

        self.depth_map_filled = np.zeros(self.depth_map_coarse.shape, dtype=np.uint8)

        self.classifyPoints()

        for i in range(len(self.boundery_pts)):
            x = self.boundery_pts[i][0]
            y = self.boundery_pts[i][1]
            self.filled_pts.append((x, y))
            for j in range(0, 3):
                self.depth_map_filled[x, y, j] = self.depth_map_coarse[x, y, j]

        #cv.imshow('depth map', self.depth_map_filled)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        max_itr = 100

        height = self.depth_map_coarse.shape[0]
        width = self.depth_map_coarse.shape[1]

        for i in range(max_itr):
            add_count = 0

            #fine new points
            down_pt = set([((i[0] + 1), i[1]) for i in self.filled_pts if (i[0] + 1) < height]) - set(self.filled_pts)
            up_pt = set([((i[0] - 1), i[1]) for i in self.filled_ptsif (i[0] - 1) > -1]) - set(self.filled_pts)
            right_pt = set([(i[0], (i[1] - 1)) for i in self.filled_pts if (i[1] + 1) < width]) - set(self.filled_pts)
            left_pt = set([(i[0], (i[1] + 1)) for i in self.filled_pts if (i[1] - 1) > -1]) - set(self.filled_pts)
            inner_pt = set([(i[0], (i[1])) for i in self.inner_pts])

            up_left = up_pt.intersection(left_pt)
            up_right = up_pt.intersection(right_pt)
            up = up_left.union(up_right)

            down_left = down_pt.intersection(left_pt)
            down_right = down_pt.intersection(right_pt)
            down = down_left.union(down_right)

            new_pt = up.union(down)

            new_pt = list(new_pt.intersection(inner_pt))

            print(len(new_pt))

            for i in range(len(new_pt)):

                x = new_pt[i][0]
                y = new_pt[i][1]

                if (((x, y + 1) in self.filled_pts) or ((x, y-1) in self.filled_pts)) and (((x+1, x) in self.filled_pts) or ((x-1, y) in self.filled_pts)):
                    depth = self.solve_depth(x, y)
                    self.filled_pts.append((x, y))
                    add_count += 1
                    for j in range(0, 3):
                        self.depth_map_filled[x, y, j] = depth

                else:
                    continue


            print('fill depth map, add %d' % ( add_count))

        cv.imshow('depth map', self.depth_map_filled)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    depth_map = DepthMap(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\smpl_depth_map.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\normal.jpeg')

    depth_map.warp_depth_map()