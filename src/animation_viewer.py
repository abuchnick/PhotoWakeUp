import moderngl_window as mglw


class AnimationWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1280, 720)
    resizable = False

    i = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def render(self, time, frametime):
        self.ctx.clear(1.0, 0.0, 0.0)

    # def mouse_position_event(self, x, y, dx, dy):
    #     print("Mouse position:", x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        print("Mouse drag:", x, y, dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        print("Mouse wheel:", x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        print("Mouse button {} pressed at {}, {}".format(button, x, y))

    def mouse_release_event(self, x: int, y: int, button: int):
        print("Mouse button {} released at {}, {}".format(button, x, y))


def run_animation():
    mglw.run_window_config(AnimationWindow)


if __name__ == '__main__':
    run_animation()
