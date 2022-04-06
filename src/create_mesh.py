import numpy as np
import cv2 as cv
import trimesh
from itertools import chain


class Reconstruct:

    def __init__(self, mask,  depth_front, depth_back, projection_matrix):
        self.depth_front = depth_front  # instead of path to images, we just give the images
        self.depth_back = depth_back
        self.mask = mask
        self.inner = []
        self.boundary = []
        self.thresholded_mask = None
        self.projection_matrix_inv = np.linalg.inv(projection_matrix)

    def classify_points(self):
        gray_mask = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)  # to use cv.threshold the mask must be a grayscale mask
        _, self.thresholded_mask = cv.threshold(gray_mask, 100, 255, cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, hierarchy = cv.findContours(self.thresholded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        outer_contour = contours[0]
        inner_contours = []
        i = hierarchy[0][0][2]
        while True:
            inner_contours.append(contours[i])
            i = hierarchy[0][i][0]
            if i == -1:
                break

        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                if cv.pointPolygonTest(outer_contour, (j, i), False) == 1 \
                        and all([cv.pointPolygonTest(c, (j, i), False) == -1 for c in inner_contours]):
                    self.inner.append([i, j])
                elif cv.pointPolygonTest(outer_contour, (j, i), False) == 0 \
                        or any([cv.pointPolygonTest(c, (j, i), False) == 0 for c in inner_contours]):
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

            if q1 in [np.Inf, np.NINF]:
                print(f"{q1=} {x=} {y=}")
            if q2 in [np.Inf, np.NINF]:
                print(f"{q2=} {x=} {y=}")

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
            if self.thresholded_mask[y+1, x+1] > 0:
                if self.thresholded_mask[y+1, x] > 0:
                    faces.append([map_front[(y, x)], map_front[(y + 1, x + 1)], map_front[(y + 1, x)]])
                    faces.append([map_back[(y, x)], map_back[(y + 1, x)], map_back[(y + 1, x + 1)]])
                if self.thresholded_mask[y, x+1] > 0:
                    faces.append([map_front[(y, x)], map_front[(y, x + 1)], map_front[(y + 1, x + 1)]])
                    faces.append([map_back[(y, x)], map_back[(y + 1, x + 1)], map_back[(y, x + 1)]])
            else:
                if self.thresholded_mask[y+1, x] > 0\
                        and self.thresholded_mask[y, x+1] > 0:
                    faces.append([map_front[(y, x)], map_front[(y, x + 1)], map_front[(y + 1, x)]])
                    faces.append([map_back[(y, x)], map_back[(y + 1, x)], map_back[(y, x + 1)]])
            if x > 0 and self.thresholded_mask[y, x - 1] == 0 \
                    and self.thresholded_mask[y + 1, x - 1] > 0 \
                    and self.thresholded_mask[y + 1, x] > 0:
                faces.append([map_front[(y, x)], map_front[(y + 1, x)], map_front[(y + 1, x - 1)]])
                faces.append([map_back[(y, x)], map_back[(y + 1, x - 1)], map_back[(y + 1, x)]])

        # need to apply uv coords
        faces = np.array(faces)
        vertices = np.array(vertices)
        vertices = vertices / np.array([w/2, h/2, 1]) - np.array([1, 1, 0])

        uv_coords = np.take(a=vertices, indices=faces, axis=0)

        homogenous_vertices = np.c_[vertices, np.ones(vertices.shape[0])]
        transformed_vertices = np.einsum('ij, vj->vi', self.projection_matrix_inv, homogenous_vertices)

        transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3:]
        #mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces)
        # trimesh.smoothing.filter_laplacian(mesh)
        #mesh.show()

        return transformed_vertices, faces, uv_coords


if __name__ == "__main__":
    _mesh = Reconstruct(
        cv.imread(r'./images/smpl_mask.jpg'),
        cv.imread(r'./images/smpl_front.tiff', cv.IMREAD_ANYDEPTH),
        cv.imread(r'./images/smpl_back.tiff', cv.IMREAD_ANYDEPTH),
        np.load(r'./projection_matrix.npy'))
    _mesh.create_mesh()
