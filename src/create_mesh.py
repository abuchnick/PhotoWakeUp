import numpy as np
import cv2 as cv
import trimesh
from inverse_warp import get_contours


class Reconstruct:

    def __init__(self, mask,  depth_front, depth_back, projection_matrix):
        self.depth_front = depth_front  # instead of path to images, we just give the images
        self.depth_back = depth_back
        self.mask = mask
        self.inner = []
        self.boundary = []
        self.projection_matrix_inv = np.linalg.inv(projection_matrix)

    def classify_points(self):
        contours = get_contours(self.mask)
        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if cv.pointPolygonTest(contours, (j, i), False) == 1:
                    self.inner.append([i, j])
                elif cv.pointPolygonTest(contours, (j, i), False) == 0:
                    self.boundary.append([i, j])

    @staticmethod
    def uv_coordinates(_vertices, _faces, _img_size):
        img_width = _img_size[1]
        img_height = _img_size[0]
        normalized_xy_coords = _vertices[:2] / np.array([img_width, img_height])
        return np.take(a=normalized_xy_coords, indices=_faces, axis=0)  # shape(m, 3, 2)

    def create_mesh(self):
        self.classify_points()

        h = self.mask.shape[0]
        w = self.mask.shape[1]
        map_front = {}
        map_back = {}
        points = self.inner + self.boundary

        vertices = []
        faces = []

        for idx, i in enumerate(self.inner):
            q1 = self.depth_front[i[0], i[1]]
            q2 = self.depth_back[i[0], i[1]]
            if q1 > q2:
                mid = (q1 + q2) / 2
                q1 = mid
                q2 = mid
            vertices.append([i[1], i[0], q1])
            vertices.append([i[1], i[0], q2])
            map_front[(i[0], i[1])] = 2 * idx
            map_back[(i[0], i[1])] = 2 * idx + 1

        len_pts = len(self.inner)
        for idx, i in enumerate(self.boundary):
            q1 = self.depth_front[i[0], i[1]]
            q2 = self.depth_back[i[0], i[1]]
            mid = (q1 + q2) / 2
            vertices.append([i[1], i[0], mid])
            vertices.append([i[1], i[0], mid])
            map_front[(i[0], i[1])] = 2 * idx + len_pts
            map_back[(i[0], i[1])] = 2 * idx + 1 + len_pts

        for y, x in points:
            if [y + 1, x + 1] in points:
                if [y + 1, x] in points:
                    faces.append([map_front[y, x], map_front[y + 1, x], map_front[y + 1, x + 1]])
                    faces.append([map_back[y, x], map_back[y + 1, x + 1], map_back[y + 1, x]])
                if [y, x + 1] in points:
                    faces.append([map_front[y, x], map_front[y + 1, x + 1], map_front[y, x + 1]])
                    faces.append([map_back[y, x], map_back[y, x + 1], map_back[y + 1, x + 1]])

        uv_coords = self.uv_coordinates(vertices, faces, (h, w))

        # need to apply uv coords
        vertices = np.array(vertices)
        faces = np.array(faces)
        homogenous_vertices = np.c_[vertices, np.ones(self.vertices.shape[0])]
        transformed_vertices = np.einsum('ij, vj->vi', self.projection_matrix_inv, homogenous_vertices)
        transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3:]
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces)
        mesh.show()

        return transformed_vertices, faces


if __name__ == "__main__":
    pass
    # _mesh = Reconstruct(r'C:\Users\talha\Desktop\study\semester 7\inner.jpeg',
    #                     r'C:\Users\talha\Desktop\study\semester 7\depth.png',
    #                     r'C:\Users\talha\Desktop\study\semester 7\depth.png')
    # _mesh.create_mesh()
