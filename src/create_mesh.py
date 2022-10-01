import numpy as np
import cv2 as cv
import trimesh
from itertools import chain

from Image_renderer import Renderer


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
        gray_mask = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY) if len(self.mask.shape) > 2 else self.mask  # to use cv.threshold the mask must be a grayscale mask
        _, self.thresholded_mask = cv.threshold(gray_mask, 100, 1, cv.THRESH_BINARY)  # each value below 100 will become 0, and above will become 255
        contours, hierarchy = cv.findContours(self.thresholded_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # outer_contours = [contours[0]]
        # inner_contours = []
        # i = hierarchy[0][0][0]
        # while True:
        #     if i == -1:
        #         break
        #     outer_contours.append(contours[i])
        #     i = hierarchy[0][i][0]
        #     j = hierarchy[0][i][2]
        #     while j != -1:
        #         inner_contours.append(contours[j])
        #         j = hierarchy[0][j][0]

        for i in range(self.mask.shape[0]):
            for j in range(self.mask.shape[1]):
                # if any([cv.pointPolygonTest(c, (j, i), False) == 0 for c in contours]):
                #     self.boundary.append([i, j])
                # if self.thresholded_mask[i, j] == 1 \
                #         and all([cv.pointPolygonTest(c, (j, i), False) == -1 for c in inner_contours]):
                #     self.inner.append([i, j])
                # elif any([cv.pointPolygonTest(c, (j, i), False) == 1 for c in contours]):
                #     self.boundary.append([i, j])
                # else:
                #     self.thresholded_mask[i, j] = 0
                if self.thresholded_mask[i, j] == 1:
                    if np.all(np.array([
                        self.thresholded_mask[np.max((0, i-1)), j],
                        self.thresholded_mask[np.min((self.thresholded_mask.shape[0], i+1)), j],
                        self.thresholded_mask[i, np.max((0, j-1))],
                        self.thresholded_mask[i, np.min((self.thresholded_mask.shape[1], j+1))],
                    ]) == 1):
                        self.inner.append([i, j])
                    else:
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
                print(f"{q1} {x} {y}")
            if q2 in [np.Inf, np.NINF]:
                print(f"{q2} {x} {y}")

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
                    faces.append([map_front[(y, x)], map_front[(y + 1, x)], map_front[(y + 1, x + 1)]])
                    faces.append([map_back[(y, x)], map_back[(y + 1, x + 1)], map_back[(y + 1, x)], ])
                if self.thresholded_mask[y, x+1] > 0:
                    faces.append([map_front[(y, x)], map_front[(y + 1, x + 1)], map_front[(y, x + 1)]])
                    faces.append([map_back[(y, x)], map_back[(y, x + 1)], map_back[(y + 1, x + 1)]])
            else:
                if self.thresholded_mask[y+1, x] > 0\
                        and self.thresholded_mask[y, x+1] > 0:
                    faces.append([map_front[(y, x)], map_front[(y + 1, x)], map_front[(y, x + 1)], ])
                    faces.append([map_back[(y, x)], map_back[(y, x + 1)], map_back[(y + 1, x)]])
            if x > 0 and self.thresholded_mask[y, x - 1] == 0 \
                    and self.thresholded_mask[y + 1, x - 1] > 0 \
                    and self.thresholded_mask[y + 1, x] > 0:
                faces.append([map_front[(y, x)], map_front[(y + 1, x - 1)], map_front[(y + 1, x)]])
                faces.append([map_back[(y, x)], map_back[(y + 1, x)], map_back[(y + 1, x - 1)]])

        # need to apply uv coords
        faces = np.array(faces, dtype=np.int32)
        vertices = np.array(vertices)
        vertices = vertices / np.array([w/2, -h/2, 1]) - np.array([1, -1, 0])

        uv_coords = (np.take(a=vertices, indices=faces, axis=0)+1)/2

        homogenous_vertices = np.c_[vertices, np.ones(vertices.shape[0])]
        transformed_vertices = np.einsum('ij, vj->vi', self.projection_matrix_inv, homogenous_vertices)

        transformed_vertices = transformed_vertices[:, :3] / transformed_vertices[:, 3:]
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces)
        export = trimesh.exchange.obj.export_obj(mesh)
        with open('mesh.obj', 'w') as file:
            file.write(export)
        # trimesh.smoothing.filter_laplacian(mesh)
        mesh.show()

        transformed_vertices = mesh.vertices
        faces = mesh.faces

        return {
            "transformed_vertices": transformed_vertices,
            "faces": faces,
            "uv_coords": uv_coords
        }


if __name__ == "__main__":
    import pickle as pkl
    from Image_renderer import Renderer

    with open('result.pkl', 'rb') as file:
        result = pkl.load(file)

    renderer = Renderer(
        vertices=result['mesh']['vertices'],
        faces=result['mesh']['faces'],
        img=cv.imread(r'./data/images_temp/goku.jpg'),
        camera_translation=result['camera']['translation'],
        camera_rotation=result['camera']['rotation']
    )

    depth_front = np.load(r'./depth_front_filled.npy')
    depth_back = np.load(r'./depth_back_filled.npy')
    mask = cv.imread(r'./refined_mask.png', cv.IMREAD_GRAYSCALE)
    mask[depth_front == np.inf] = 0
    mask[depth_front <= 0] = 0
    mask[depth_back == np.inf] = 0
    mask[depth_back <= 0] = 0
    _mesh = Reconstruct(
        mask,
        depth_front,
        depth_back,
        renderer.projection_matrix)
    mesh_data = _mesh.create_mesh()
    np.savez('mesh_data.npz', **mesh_data)

    del(renderer)
    # mesh_data = np.load('mesh_data.npz')

    renderer = Renderer(
        vertices=mesh_data['transformed_vertices'],
        faces=mesh_data['faces'],
        img=cv.imread(r'./data/images_temp/goku.jpg'),
        camera_translation=result['camera']['translation'],
        camera_rotation=result['camera']['rotation']
    )
    # render, _ = renderer.render_normals()
    render2 = renderer.render_texture(img=cv.imread(r'./data/images_temp/goku.jpg'))
    # cv.imshow('1', (np.flip(render, axis=2)+1.0)/2)
    cv.imshow('2', render2)
    cv.waitKey()
    cv.destroyAllWindows()
