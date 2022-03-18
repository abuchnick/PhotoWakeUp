import numpy as np
import cv2 as cv


class DepthMap:

    def __init__(self, mask, depth_map_coarse, normal_map):
        self. depth_map_coarse = cv.imread(depth_map_coarse)
        self.normal_map = cv.imread(normal_map)
        self.mask = cv.imread(mask)
        self.inner_pts = []
        self.boundery_pts = []
        self.filled_pts = []
        self.depth_map_filled = None

    def isvalidpix(self, image, x, y):
        if (image[y, x] == [0, 0, 0]).all() == True:
            return 'not inner point'
        elif (image[y, x] == [255, 255, 255]).all() == True:
            return 'inner point not filled'
        else:
            return 'inner point filled'

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

    def valid_neighbor(self, image, x, y):

        x_valid, y_valid = 0, 0

        if self.isvalidpix(image, x - 1, y) == 1:
            x_valid = x_valid + 1

        if self.isvalidpix(image, x + 1, y) == 1:
            x_valid = x_valid + 1

        if self.isvalidpix(image, x, y - 1) == 1:
            y_valid = y_valid + 1

        if self.isvalidpix(image, x, y + 1) == 1:
            y_valid = y_valid + 1

        return x_valid, y_valid


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
            h = self.boundery_pts[i][0]
            w = self.boundery_pts[i][1]
            self.filled_pts.append((h, w))
            for j in range(0, 3):
                self.depth_map_filled[h, w, j] = self.depth_map_coarse[h, w, j]

        #cv.imshow('depth map', self.depth_map_filled)
        #cv.waitKey(0)
        #cv.destroyAllWindows()

        iter_num = 100

        height = self.depth_map_coarse.shape[0]
        width = self.depth_map_coarse.shape[1]

        for iter_ in range(iter_num):
            add_count = 0
            down = set([((i[0] + 1), i[1]) for i in self.filled_pts]) - set(self.filled_pts)
            up = set([((i[0] - 1), i[1]) for i in self.filled_pts]) - set(self.filled_pts)
            right = set([(i[0], (i[1] - 1)) for i in self.filled_pts]) - set(self.filled_pts)
            left = set([(i[0], (i[1] + 1)) for i in self.filled_pts]) - set(self.filled_pts)
            inner = set([(i[0], (i[1])) for i in self.inner_pts])

            a = up.intersection(left)
            b = up.intersection(right)
            c = a.union(b)

            d = down.intersection(left)
            e = down.intersection(right)
            f = d.union(e)

            g = f.union(c)

            final = list(g.intersection(inner))





            for i in range(len(final)):
                h = final[i][0]
                w = final[i][1]

                if (h, w) not in self.filled_pts:
                    if (((h, w + 1) in self.filled_pts) or ((h, w-1) in self.filled_pts)) and (((h+1, h) in self.filled_pts) or ((h-1, w) in self.filled_pts)):
                        depth = self.solve_depth(h, w)
                        self.filled_pts.append((h, w))
                        add_count += 1
                        for j in range(0, 3):
                            self.depth_map_filled[h, w, j] = depth

                    else:
                        continue
                else:
                    print('bla')

            print('%d/%d fill depth map, add %d' % (iter_, iter_num, add_count))

        cv.imshow('depth map', self.depth_map_filled)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    depth_map = DepthMap(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\smpl_depth_map.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\normal.jpeg')

    depth_map.warp_depth_map()