import numpy as np
import cv2 as cv
import trimesh


class Reconstruct:

    def __init__(self, mask, depth_map_coarse):
        self.depth_map = cv.imread(depth_map_coarse)
        self.mask = cv.imread(mask)
        self.inner_pts = []
        self.boundary_pts = []
        self.vertices = []
        self.faces = []

    def classify_points(self):
        gray_img = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)  # to use cv.threshold the img must be a grayscale img
        threshold_used, thresholded_image = cv.threshold(gray_img, 100, 255,
                                                         cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, hierarchy = cv.findContours(thresholded_image, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # retrieve all points in contour(don't approximate) and save full hirarchy
        contours = np.array(contours[0]).squeeze(
            1)  # this will take the contours of the first object only. cast for nd-array since the output is a list, and squeeze dim 1 since its redundant

        for i in range(self.depth_map.shape[0]):
            for j in range(self.depth_map.shape[1]):
                if cv.pointPolygonTest(contours, (j, i), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv.pointPolygonTest(contours, (j, i), False) == 0:
                    self.boundary_pts.append([i, j])

    def create_mesh(self):
        self.classify_points()
        points = self.inner_pts + self.boundary_pts
        mapping = {}

        print("Creating vertices")

        for idx, point in enumerate(points):
            y, x = point
            z = self.depth_map[y, x][0]
            self.vertices.append([x, y, z])
            mapping[(x, y)] = idx

        print("Creating faces")

        for point in points:
            y, x = point
            if [y + 1, x] in points and [y + 1, x + 1] in points:
                self.faces.append([mapping[(x, y)], mapping[(x, y + 1)], mapping[(x + 1, y + 1)]])

        _mesh = trimesh.Trimesh(vertices=self.vertices,
                                faces=self.faces)
        _mesh.show()


if __name__ == "__main__":
    mesh = Reconstruct(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
                       r'C:\Users\talha\Desktop\study\semester 7\depth.png')
    mesh.create_mesh()
