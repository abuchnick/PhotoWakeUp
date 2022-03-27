import numpy as np
import cv2 as cv
import trimesh
from inverse_warp import get_contours


class Reconstruct:

    def __init__(self, mask, depth_map_coarse):
        self.depth_map = depth_map_coarse  # instead of path to images, we just give the images
        self.mask = mask
        self.inner_pts = []
        self.boundary_pts = []

    def classify_points(self):
        contours = get_contours(self.mask)
        for i in range(self.depth_map.shape[0]):
            for j in range(self.depth_map.shape[1]):
                if cv.pointPolygonTest(contours, (j, i), False) == 1:
                    self.inner_pts.append([i, j])
                elif cv.pointPolygonTest(contours, (j, i), False) == 0:
                    self.boundary_pts.append([i, j])

    def create_mesh(self):
        self.classify_points()
        points = self.inner_pts + self.boundary_pts

        vertices = []
        faces = []
        mapping = {}

        print("Creating vertices")

        for idx, point in enumerate(points):
            y, x = point
            z = self.depth_map[y, x][0]
            vertices.append([x, y, z])
            mapping[(x, y)] = idx

        print("Creating faces")

        for point in points:
            y, x = point
            if [y + 1, x] in points and [y + 1, x + 1] in points:
                faces.append([mapping[(x, y)], mapping[(x, y + 1)], mapping[(x + 1, y + 1)]])

        _mesh = trimesh.Trimesh(vertices=vertices,
                                faces=faces)
        _mesh.show()

        return np.array(vertices), np.array(faces)


if __name__ == "__main__":
    pass
    # mesh = Reconstruct(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
    #                    r'C:\Users\talha\Desktop\study\semester 7\depth.png')
    # mesh.create_mesh()
