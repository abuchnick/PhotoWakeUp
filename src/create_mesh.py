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

    @staticmethod
    def uv_coordinates(_vertices, _faces, _img_size):
        img_width = _img_size[1]
        img_height = _img_size[0]
        normalized_xy_coords = _vertices[:, :2] / np.array([img_width, img_height])
        return np.take(a=normalized_xy_coords, indices=_faces, axis=0)  # shape(m, 3, 2)

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

        # uv_coords = self.uv_coordinates(vertices, faces, (h, w))

        # need to apply uv coords
        vertices = np.array(vertices)
        vertices = vertices / np.array([w/2, h/2, 1]) - np.array([1, 1, 0])
        faces = np.array(faces)
        homogenous_vertices = np.c_[vertices, np.ones(vertices.shape[0])]
        transformed_vertices = np.einsum('ij, vj->vi', self.projection_matrix_inv, homogenous_vertices)
        # print(np.any(transformed_vertices[:, 3:] == 0))
        # print(np.any(transformed_vertices[:, 3:] == np.Inf))
        # print(np.any(transformed_vertices[:, 3:] == np.NAN))
        # print(np.any(transformed_vertices[:, 3:] == np.NINF))
        #
        # print(np.any(transformed_vertices[:, :3] == 0))
        # print(np.any(transformed_vertices[:, :3] == np.Inf))
        # print(np.any(transformed_vertices[:, :3] == np.NAN))
        # print(np.any(transformed_vertices[:, :3] == np.NINF))
        #
        # print(np.any(transformed_vertices == 0))
        # print(np.any(transformed_vertices == np.Inf))
        # print(np.any(transformed_vertices == np.NAN))
        # print(np.any(transformed_vertices == np.NINF))

        transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3:]
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces)
        mesh.show()

        return transformed_vertices, faces


if __name__ == "__main__":
    _mesh = Reconstruct(
        cv.imread(r'./images/smpl_mask.jpg'),
        cv.imread(r'./images/smpl_front.tiff', cv.IMREAD_ANYDEPTH),
        cv.imread(r'./images/smpl_back.tiff', cv.IMREAD_ANYDEPTH),
        np.load(r'C:\Users\talha\Desktop\study\semester 7/projection_matrix.npy'))
    _mesh.create_mesh()
