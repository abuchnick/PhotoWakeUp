# from concurrent.futures import process
import cv2
import moderngl as gl
import numpy as np
from typing import Tuple, Union
# import matplotlib.pyplot as plt
import pickle as pkl
# import import_smplifyx as smplifyx
import trimesh
import os
from camera import Camera

SHADERS_FOLDER = os.path.realpath(os.path.join(__file__, '../shaders'))


def map_ranges(value, from_range, to_range):
    return (value - from_range[0]) * ((to_range[1] - to_range[0]) / (from_range[1] - from_range[0])) + to_range[0]


class Renderer:
    def __init__(
        self,
        vertices: np.ndarray,  # (n, 3)
        faces: np.ndarray,  # (m, 3) indices,
        img: np.ndarray,
        camera_translation: np.ndarray,  # (3,)
        camera_rotation: np.ndarray,  # (3, 3)
        # camera_center=(0., 0.),  # (2,)
        focal_length=5000.0,
        znear=1.0,
        zfar=50.0
    ):
        self.ctx = gl.create_context(standalone=True)
        self.ctx.enable(gl.DEPTH_TEST)
        self.vertices = vertices
        self.faces = faces
        self.normals = None
        self.normal_depth_rescale = None
        self.znear = znear
        self.zfar = zfar
        self.img = img
        self.img_shape = img.shape
        camera = Camera(
            camera_translation=camera_translation,
            rotation_matrix=camera_rotation,
            znear=znear,
            zfar=zfar,
            focal_length=focal_length)

        self.projection_matrix = camera.matrix(self.img_shape)

        self.normals_projection = camera.get_transform_matrix()[0:3, 0:3]

        self.vbo = self.ctx.buffer(
            data=vertices.astype(np.float32).tobytes()
        )
        self.ibo = self.ctx.buffer(data=faces.astype(np.uint16).tobytes())

        self.vnbo = self.ctx.buffer(reserve=self.vbo.size)

        self.fbo = self.ctx.simple_framebuffer(
            size=tuple(reversed(self.img_shape)),
            components=3
        )

    def __del__(self):
        self.fbo.release()
        self.vnbo.release()
        self.ibo.release()
        self.vbo.release()
        self.ctx.release()

    def render_solid(self, get_depth_map=False, back_side=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        program = self.load_shader('solid')

        R = np.eye(4, dtype=self.projection_matrix.dtype)
        R[2, 2] = -1 if back_side else 1

        program['projection'].write(
            (R @ self.projection_matrix).tobytes('F')
        )

        vao = self.ctx.vertex_array(
            program=program,
            content=[(self.vbo, '3f4', 'vertex')],
            index_buffer=self.ibo,
            index_element_size=2
        )

        self.fbo.use()
        self.fbo.clear()
        vao.render()

        render = np.flip(np.frombuffer(
            self.fbo.read(), dtype=np.uint8).reshape(*self.img_shape,  3), axis=[0, 2])

        vao.release()
        program.release()

        if get_depth_map:
            depth_map = np.flip(np.frombuffer(
                self.fbo.read(attachment=-1, dtype='f4'), dtype='f4').reshape(*self.img_shape), axis=0)
            depth_map = np.where(depth_map == 1.0, float('inf'),  1.0-2.0*depth_map if back_side else 2.0*depth_map-1.0)
            return render, depth_map

        return render

    def render_normals(self, back_side=False) -> Tuple[np.ndarray, np.ndarray]:
        program = self.load_shader('normals')

        P = self.projection_matrix
        if back_side:
            P = P * np.array([1, 1, -1, 1], dtype=np.float32).reshape(4, 1)

        program['projection'].write(P.tobytes('F'))

        program['normals_projection'].write(
            np.diag(np.array([-1, -1, (-1 if back_side else 1)], dtype=np.float32)).tobytes('F')
        )

        if self.normals is None:
            N = self.vertices.shape[0]
            h, w = self.img_shape
            homogeneus_vertices = np.c_[self.vertices, np.ones(N)]
            homogeneus_projected_vertices = np.einsum('ij,vj->vi', P, homogeneus_vertices)
            projected_vertices = homogeneus_projected_vertices[:, :3] / homogeneus_projected_vertices[:, [3]]
            zmin, zmax = projected_vertices[:, 2].min(), projected_vertices[:, 2].max()
            self.normal_depth_rescale = np.array([[zmin, zmax], [0., 500.]])
            # remaping x,y to their pixel values in image (for integration) and z to the range (0,500) for numerical stability
            projected_vertices = (projected_vertices + np.array([1., 1., -zmin])) * np.array([w / 2., h / 2., 500./(zmax-zmin)])
            mesh = trimesh.Trimesh(
                vertices=projected_vertices,
                faces=self.faces,
                process=False
            )
            self.normals = mesh.vertex_normals * np.array([1, -1, 1])
            self.vnbo.write(data=self.normals.astype(np.float32).tobytes())

        vao = self.ctx.vertex_array(
            program=program,
            content=[
                (self.vbo, '3f4', 'vertex'),
                (self.vnbo, '3f4', 'normal')
            ],
            index_buffer=self.ibo,
            index_element_size=2
        )

        fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(size=tuple(reversed(self.img_shape)), components=3, dtype='f4')],
            depth_attachment=self.ctx.depth_renderbuffer(size=tuple(reversed(self.img_shape)))
        )
        fbo.use()
        fbo.clear()
        vao.render()

        render = np.flip(np.frombuffer(
            fbo.read(dtype='f4'), dtype=np.float32).reshape(*self.img_shape,  3), axis=0)

        fbo.release()
        vao.release()
        program.release()

        return render, self.normal_depth_rescale

    def render_skinning_map(self, skinning_map) -> np.ndarray:
        program = self.load_shader('skinning_map')

        program['projection'].write(
            (self.projection_matrix).tobytes('F')
        )

        skinning_weights_buffer = self.ctx.buffer(reserve=skinning_map.shape[0]*4)

        vao = self.ctx.vertex_array(
            program=program,
            content=[
                (self.vbo, '3f4', 'vertex'),
                (skinning_weights_buffer, 'f4', 'weight')
            ],
            index_buffer=self.ibo,
            index_element_size=2
        )

        fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(size=tuple(reversed(self.img_shape)), components=1, dtype='f4')],
            depth_attachment=self.ctx.depth_renderbuffer(size=tuple(reversed(self.img_shape)))
        )
        fbo.use()
        skinning_map_per_joint = []
        for i in range(22):  # we only use the first 22 weight since we dont animate face and hands
            fbo.clear()
            skinning_weights_buffer.write(skinning_map[:, i].astype(np.float32).tobytes())
            vao.render()

            render = np.flip(np.frombuffer(
                fbo.read(components=1, dtype='f4'), dtype=np.float32).reshape(*self.img_shape, 1), axis=0)
            skinning_map_per_joint.append(render)

        fbo.release()
        vao.release()
        skinning_weights_buffer.release()
        program.release()
        return np.concatenate(skinning_map_per_joint, axis=-1)

    def load_shader(self, shader_name):
        with open(os.path.join(SHADERS_FOLDER, shader_name + '_vertex.glsl'), 'r') as file:
            vertex_shader = file.read()
        with open(os.path.join(SHADERS_FOLDER, shader_name + '_fragment.glsl'), 'r') as file:
            fragment_shader = file.read()
        return self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

    def get_uv_coords(self):
        num_vertices = self.vertices.shape[0]
        hom_coords = np.concatenate((self.vertices, np.ones((num_vertices, 1))), axis=1)
        projected_coords = (self.projection_matrix @ hom_coords.T).T
        return (projected_coords[:, :2] / projected_coords[:, [3]] + 1) / 2
    
    def render_texture(self, img, vertices = None):
        UVs = self.get_uv_coords()
        program = self.load_shader("animation")
        
        if vertices:
            vbo = self.ctx.buffer(
            data=vertices.astype(np.float32).tobytes()
        )
        else:
            vbo = self.vbo
        
        program['projection'].write(
            (self.projection_matrix).tobytes('F')
        )

        uvbo = self.ctx.buffer(
            data=UVs.astype(np.float32).tobytes()
        )

        vao = self.ctx.vertex_array(
            program=program,
            content=[
                (vbo, '3f4', 'vertex'),
                (uvbo, '2f4', 'uv_coordinates')
            ],
            index_buffer=self.ibo,
            index_element_size=2
        )
        
        
        texture = self.ctx.texture(self.img_shape, components=3, data=np.flip(img, axis=[0, 2]).tobytes())
        texture.use(0)
        program['Texture'] = 0
        
        self.fbo.use()
        self.fbo.clear()
        vao.render()

        render = np.flip(np.frombuffer(
            self.fbo.read(), dtype=np.uint8).reshape(*self.img_shape,  3), axis=[0, 2])

        vao.release()
        program.release()
        texture.release()
        
        return render


if __name__ == '__main__':
    with open('result.pkl', 'rb') as file:
        result = pkl.load(file)['weird'][0]
    vertices = result['mesh']['vertices']
    img = cv2.imread("images/weird.jpeg")
    renderer = Renderer(
        vertices=result['mesh']['vertices'],
        faces=result['mesh']['faces'],
        img=img,
        camera_translation=result['camera']['translation'],
        camera_rotation=result['camera']['rotation']
    )

    mask, depth_front = renderer.render_solid(get_depth_map=True, back_side=False)
    _, depth_back = renderer.render_solid(get_depth_map=True, back_side=True)
    normals_front, rescale_front = renderer.render_normals(back_side=False)
    normals_back, rescale_back = renderer.render_normals(back_side=True)

    skinning_map = renderer.render_skinning_map(result['mesh']['skinning_map'])

    render = renderer.render_texture(img)
    cv2.imshow("ahlan itzik", render)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(1)
    
    # for i in range(skinning_map.shape[-1]):
    #     cv2.imshow('skinning_map', skinning_map[:, :, i])
    #     if cv2.waitKey(200) != -1:
    #         break
    # cv2.destroyAllWindows()

    dmin = np.min(depth_front)
    dmax = np.max(np.where(depth_front == np.max(depth_front), float('-inf'), depth_front))

    cv2.imshow('render1', mask)
    cv2.imshow('render2', (np.flip(normals_front, axis=2)+1.0)/2)
    cv2.imshow('depth_front', 1. - (depth_front - dmin) / (dmax-dmin))
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    cv2.imwrite("smpl_depth_front.tiff", depth_front)
    cv2.imwrite("smpl_depth_back.tiff", depth_back)
    cv2.imwrite("smpl_mask.jpg", mask)
    np.savez('smpl_normals_front.npz', normals=normals_front, rescale=rescale_front)
    np.savez('smpl_normals_back.npz', normals=normals_back, rescale=rescale_back)
    np.save('projection_matrix.npy', renderer.projection_matrix)
    np.save('skinning_map.npy', skinning_map)
