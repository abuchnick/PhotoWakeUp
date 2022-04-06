from turtle import distance
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class Camera():
    def __init__(self, camera_translation=(0, 0, 0), rotation_matrix=None, distance=1.0, znear=1.0, zfar=50.0, focal_length=5000.0):
        self.camera_translation = np.squeeze(camera_translation)
        self.rotation_matrix = np.eye(3) if rotation_matrix is None else rotation_matrix
        self.znear = znear
        self.zfar = zfar
        self.focal_length = focal_length
        self.target = self.camera_translation + distance * self.rotation_matrix[:, 2]

    def get_projection_matrix(self, img_shape):
        width, height = float(img_shape[1]), float(img_shape[0])
        # cx, cy = float(camera_center[0]), float(camera_center[1])
        focal = float(self.focal_length)
        P = np.zeros((4, 4), dtype=np.float32)
        P[0][0] = 2. * focal / width
        P[1][1] = 2. * focal / height
        # P[0][2] = 2. * cx / width
        # P[1][2] = 2. * cy / height
        P[3][2] = -1.

        zn, zf = self.znear, self.zfar
        if zf is None:
            P[2][2] = -1.0
            P[2][3] = -2.0 * zn
        else:
            P[2][2] = (zf + zn) / (zn - zf)
            P[2][3] = (2 * zf * zn) / (zn - zf)
        return P

    def get_transform_matrix(self):
        translation = np.copy(self.camera_translation)
        # Equivalent to 180 degrees around the y-axis. Transforms the fit to
        # OpenGL compatible coordinate system.
        translation[0] *= -1.0
        rotation = np.copy(self.rotation_matrix)
        transpose_rotation = np.transpose(rotation, (1, 0))
        V = np.eye(4, dtype=np.float32)
        V[0:3, 0:3] = transpose_rotation
        V[:3, 3] = -1 * translation
        return V

    def matrix(self, img_shape):
        return self.get_projection_matrix(img_shape) @ self.get_transform_matrix()

    def orbit(self, dx, dy):
        distance = np.linalg.norm(self.target - self.camera_translation)
        self.rotation_matrix = self.rotation_matrix
        Y = R.from_rotvec(np.array([0, dy, 0])).as_matrix()
        self.target

    def pan(self, dx, dy):
        movement = (dx/1000.) * self.rotation_matrix[:, 0] + (dy/1000.) * self.rotation_matrix[:, 1]
        self.camera_translation += movement
        self.target += movement

    def slide(self, dz):
        movement = (dz*1000.) * self.rotation_matrix[:, 2] / self.focal_length
        self.camera_translation += movement
