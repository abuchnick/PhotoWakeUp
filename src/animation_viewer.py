import moderngl_window as mglw
from camera import Camera
import pickle as pkl
import os
import import_smplifyx as smplifyx
import numpy as np


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with open(os.path.join('src', 'shaders', 'solid_vertex.glsl'), 'r') as file:
            vertex_shader = file.read()
        with open(os.path.join('src', 'shaders', 'solid_fragment.glsl'), 'r') as file:
            fragment_shader = file.read()
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        self.vbo = self.ctx.buffer(
            data=self.mesh['vertices'].astype(np.float32).tobytes()
        )
        self.ibo = self.ctx.buffer(data=self.mesh['faces'].astype(np.uint16).tobytes())

        # self.vbo = self.ctx.buffer(
        #     data=VERTS.astype(np.float32).tobytes()
        # )
        # self.ibo = self.ctx.buffer(data=FACES.astype(np.uint16).tobytes())

        self.vao = self.ctx.vertex_array(
            program=self.program,
            content=[(self.vbo, '3f4', 'vertex')],
            index_buffer=self.ibo,
            index_element_size=2
        )

    # @classmethod
    # def add_arguments(cls, parser: ArgumentParser):
    #     parser.add_argument()

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
    AnimationWindow.mesh = data['mesh']
    AnimationWindow.window_size = (1000, 1200)
    AnimationWindow.camera = Camera(
        camera_translation=data['camera']['translation'],
        rotation_matrix=data['camera']['rotation'],
        znear=1.0,
        zfar=50.0,
        focal_length=5000.0
    )
    mglw.run_window_config(AnimationWindow)
