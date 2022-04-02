import numpy as np
import cv2 as cv
import trimesh
from inverse_warp import get_contours
from itertools import chain


class Reconstruct:

    def __init__(self, mask,  depth_front, depth_back, projection_matrix):
        self.depth_front = depth_front  # instead of path to images, we just give the images
        self.depth_back = depth_back
        self.mask = mask
        self.inner = []
        self.boundary = []
        self.contours = None
        self.projection_matrix_inv = np.linalg.inv(projection_matrix)

    def classify_points(self):
        self.contours = get_contours(self.mask)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if cv.pointPolygonTest(self.contours, (j, i), False) == 1:
                    self.inner.append([i, j])
                elif cv.pointPolygonTest(self.contours, (j, i), False) == 0:
                    self.boundary.append([i, j])

    def create_mesh(self):

        self.classify_points()

        h = self.mask.shape[0]
        w = self.mask.shape[1]
        map_front = {}
        map_back = {}

        vertices = []
        faces = []
        idx = 0

        for y, x in self.inner:
            q1 = self.depth_front[y, x]
            q2 = self.depth_back[y, x]

            if q1 > q2:
                mid = (q1 + q2) / 2
                q1 = mid
                q2 = mid

            vertices.append([x, y, q1])
            vertices.append([x, y, q2])
            map_front[(y, x)] = idx
            idx += 1
            map_back[(y, x)] = idx
            idx += 1

        for y, x in self.boundary:
            q1 = self.depth_front[y, x]
            q2 = self.depth_back[y, x]

            mid = (q1 + q2) / 2
            vertices.append([x, y, mid])
            vertices.append([x, y, mid])
            map_front[(y, x)] = idx
            idx += 1
            map_back[(y, x)] = idx
            idx += 1

        for y, x in chain(self.inner, self.boundary):
            if cv.pointPolygonTest(self.contours, (x+1, y+1), False) >= 0:
                if cv.pointPolygonTest(self.contours, (x, y+1), False) >= 0:
                    faces.append([map_front[(y, x)], map_front[(y + 1, x + 1)], map_front[(y + 1, x)]])
                    faces.append([map_back[(y, x)], map_back[(y + 1, x)], map_back[(y + 1, x + 1)]])
                if cv.pointPolygonTest(self.contours, (x+1, y), False) >= 0:
                    faces.append([map_front[(y, x)], map_front[(y, x + 1)], map_front[(y + 1, x + 1)]])
                    faces.append([map_back[(y, x)], map_back[(y + 1, x + 1)], map_back[(y, x + 1)]])

        # need to apply uv coords
        faces = np.array(faces)
        vertices = np.array(vertices)
        vertices = vertices / np.array([w/2, h/2, 1]) - np.array([1, 1, 0])

        uv_coords = np.take(a=vertices, indices=faces, axis=0)

        homogenous_vertices = np.c_[vertices, np.ones(vertices.shape[0])]
        transformed_vertices = np.einsum('ij, vj->vi', self.projection_matrix_inv, homogenous_vertices)

        transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3:]
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces)
        mesh.show()

        return transformed_vertices, faces, uv_coords


if __name__ == "__main__":
    _mesh = Reconstruct(
        cv.imread(r'./images/smpl_mask.jpg'),
        cv.imread(r'./images/smpl_front.tiff', cv.IMREAD_ANYDEPTH),
        cv.imread(r'./images/smpl_back.tiff', cv.IMREAD_ANYDEPTH),
        np.load(r'C:\Users\talha\Desktop\study\semester 7/projection_matrix.npy'))
    _mesh.create_mesh()
