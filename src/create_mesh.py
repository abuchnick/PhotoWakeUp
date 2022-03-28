import numpy as np
import cv2 as cv
import trimesh


class Reconstruct:

    def __init__(self, mask, depth_front, depth_back):
        self.depth_front = cv.imread(depth_front, cv.IMREAD_ANYDEPTH)
        self.depth_back = cv.imread(depth_back, cv.IMREAD_ANYDEPTH)
        self.mask = cv.imread(mask)
        self.inner = []
        self.boundary = []

    def classify_points(self):
        gray_img = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)  # to use cv.threshold the img must be a grayscale img
        threshold_used, thresholded_image = cv.threshold(gray_img, 100, 255,
                                                         cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
        contours = np.array(contours[0]).squeeze(
            1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant

        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if cv.pointPolygonTest(contours, (j, i), False) == 1:
                    self.inner.append([i, j])
                elif cv.pointPolygonTest(contours, (j, i), False) == 0:
                    self.boundary.append([i, j])


    def create_mesh(self):

        self.classify_points()

        h = self.mask.shape[0]
        w = self.mask.shape[1]
        map_front = {}
        map_back = {}
        qurt = []
        points = self.inner + self.boundary

        vertices = []
        faces = []

        for idx, i in enumerate(self.inner):
            q1 = -self.depth_front[i[0], i[1]]
            q2 = self.depth_back[i[0], i[1]]
            if q1 > q2:
                mid = (q1 + q2) / 2
                q1 = mid
                q2 = mid
            vertices.append([i[0], i[1], q1])
            vertices.append([i[0], i[1],q2])
            map_front[(i[1], i[0])] = 2 * idx
            map_back[(i[1], i[0])] = 2 * idx + 1

        len_pts = len(self.inner)
        for idx, i in enumerate(self.boundary):
            q1 = self.depth_front[i[0], i[1]]
            q2 = self.depth_back[i[0], i[1]]
            mid = (q1 + q2) / 2
            vertices.append([i[0], i[1], mid])
            vertices.append([i[0], i[1], mid])
            map_front[(i[1], i[0])] = 2 * idx + len_pts
            map_back[(i[1], i[0])] = 2 * idx + 1 + len_pts

        for i, j in points:
            if [j, i] in points and [j + 1] in points and [j + 1][i + 1] in points and [j][i + 1] in points:
                faces.append([map_front[j, i], map_front[j + 1, i], map_front[j, i + 1]])
                faces.append([map_front[j + 1, i], map_front[j + 1, i + 1], map_front[j, i + 1]])
                faces.append([map_back[j, i], map_back[j, i + 1], map_back[j + 1, i]])
                faces.append([map_back[j + 1, i], map_back[j, i + 1], map_back[j + 1, i + 1]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.show()

if __name__ == "__main__":
    mesh = Reconstruct(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
                 r'C:\Users\talha\Desktop\study\semester 7\depth.png',
                       r'C:\Users\talha\Desktop\study\semester 7\depth.png')

    mesh.create_mesh()