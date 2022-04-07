import moderngl_window as mglw
import moderngl as gl
from camera import Camera
import pickle as pkl
import os
import import_smplifyx as smplifyx
import numpy as np
import cv2
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19], dtype=np.int8)


class AnimationWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1280, 720)
    resizable = True
    focal_length = 5000.0
    aspect_ratio = None
    fovy = None
    znear = 1.
    zfar = 50.
    mesh = None
    button = None
    camera = None
    img = None
    # rotations_matrices = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(gl.CULL_FACE | gl.DEPTH_TEST)
        with open(os.path.join('src', 'shaders', 'animation_vertex.glsl'), 'r') as file:
            vertex_shader = file.read()
        with open(os.path.join('src', 'shaders', 'animation_fragment.glsl'), 'r') as file:
            fragment_shader = file.read()
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        self.vbo = self.ctx.buffer(
            data=self.mesh['vertices'].astype(np.float32).tobytes()
        )
        self.ibo = self.ctx.buffer(data=self.mesh['faces'].astype(np.uint16).tobytes())

        N = self.mesh['vertices'].shape[0]
        projected_verts = np.einsum('ij,vj->vi', self.camera.matrix((self.wnd.height, self.wnd.width)), np.c_[self.mesh['vertices'], np.ones(N)])
        UVs = (projected_verts[:, :2] / projected_verts[:, [3]] + 1) / 2

        self.uvbo = self.ctx.buffer(
            data=UVs.astype(np.float32).tobytes()
        )

        self.vao = self.ctx.vertex_array(
            program=self.program,
            content=[
                (self.vbo, '3f4', 'vertex'),
                (self.uvbo, '2f4', 'uv_coordinates')
            ],
            index_buffer=self.ibo,
            index_element_size=2
        )

        # batch_rigid_transform(self.rotations_matrices, mesh['unposed_joints'], PARENTS, dtype=np.float32)

        self.texture = self.ctx.texture(self.window_size, components=3, data=np.flip(self.img, axis=[0, 2]).tobytes())
        self.texture.use(0)
        self.program['Texture'] = 0

    def mouse_drag_event(self, x: int, y: int, dx, dy):
        if self.button == 1:
            self.camera.orbit(dx, dy)
        if self.button == 2:
            self.camera.pan(dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.camera.slide(y_offset)

    def resize(self, width: int, height: int):
        pass

    def render(self, time, frametime):
        self.ctx.clear(0.0, 0.0, 0.0)
        self.program['projection'].write(
            (self.camera.matrix((self.wnd.height, self.wnd.width))).tobytes('F')
        )

        self.vao.render()

    # def mouse_position_event(self, x, y, dx, dy):
    #     print("Mouse position:", x, y, dx, dy)

    def mouse_press_event(self, x, y, button):
        self.button = button

    # def mouse_release_event(self, x: int, y: int, button: int):
    #     print("Mouse button {} released at {}, {}".format(button, x, y))


if __name__ == '__main__':
    with open('result.pkl', 'rb') as file:
        data = pkl.load(file)['goku'][0]
    image = cv2.imread('.\data\images\goku.jpg')
    AnimationWindow.img = image
    AnimationWindow.mesh = data['mesh']
    AnimationWindow.window_size = (image.shape[1], image.shape[0])
    AnimationWindow.camera = Camera(
        camera_translation=data['camera']['translation'],
        rotation_matrix=data['camera']['rotation'],
        distance=np.linalg.norm(data['mesh']['vertices'].mean(axis=0) - data['camera']['translation']),
        znear=1.0,
        zfar=50.0,
        focal_length=5000.0
    )
    # AnimationWindow.rotations_matrices = np.load('skip_to_walk_rot_mats.npy')
    print("mesh", data['mesh']['vertices'].mean(axis=0))
    mglw.run_window_config(AnimationWindow)
