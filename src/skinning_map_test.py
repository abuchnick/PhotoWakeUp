import numpy as np
from Image_renderer import Renderer
from inverse_warp import Warp
import cv2
import pickle as pkl


if __name__ == '__main__':

    with open('result.pkl', 'rb') as file:
        result = pkl.load(file)

    renderer = Renderer(
        vertices=result['mesh']['vertices'],
        faces=result['mesh']['faces'],
        img=cv2.imread(r'./data/images_temp/goku.jpg'),
        camera_translation=result['camera']['translation'],
        camera_rotation=result['camera']['rotation']
    )
    print(f"{np.all(np.abs(result['mesh']['skinning_map'].sum(axis=-1) - 1.0) < 0.01)}")

    # skinning_map_render = renderer.render_skinning_map(skinning_map=result['mesh']['skinning_map'])

    # smpl_mask = renderer.render_solid()
    # print(np.all((np.abs(skinning_map_render.sum(axis=-1) - 1.0) < 0.01) | np.any(smpl_mask < 125, axis=2)))

    # warp_func = np.load('warp.npy')
    # warp = Warp(warp_func)
    # skinning_map_warped = warp(skinning_map_render)
    # warped_mask = warp(smpl_mask)
    # print(np.all((np.abs(skinning_map_warped.sum(axis=-1) - 1.0) < 0.01) | np.any(warped_mask < 125, axis=2)))

    skinning_map_warped = np.load('skinning_image_filled.npy')

    # for i in range(22):
    #     cv2.imshow('skinning_map', skinning_map_render[:, :, i])
    #     if cv2.waitKey(200) != -1:
    #         break

    # for i in range(22):
    #     cv2.imshow('skinning_map', skinning_map_warped[:, :, i])
    #     if cv2.waitKey(200) != -1:
    #         break
    # cv2.destroyAllWindows()
    # np.save('skinning_map_image.npy', skinning_map_warped)
    # skinning_map_warped = np.load('skinning_map_image.npy')

    mesh = np.load('mesh_data.npz')

    del(renderer)
    renderer = Renderer(
        vertices=mesh['transformed_vertices'],
        faces=mesh['faces'],
        img=cv2.imread(r'./data/images_temp/goku.jpg'),
        camera_translation=result['camera']['translation'],
        camera_rotation=result['camera']['rotation']
    )

    mesh_mask = renderer.render_solid()
    print(np.all((np.abs(skinning_map_warped.sum(axis=-1) - 1.0) < 0.01) | np.any(mesh_mask < 125, axis=2)))

    problematic_pixels = ~((np.abs(skinning_map_warped.sum(axis=-1) - 1.0) < 0.01) | np.any(mesh_mask < 125, axis=2))
    for x, y in zip(*np.nonzero(problematic_pixels)):
        value = np.zeros(skinning_map_warped.shape[-1])
        for i in (x-1, x, x+1):
            for j in (y-1, y, y+1):
                value += skinning_map_warped[i, j]
        value = value / np.sum(value)
        skinning_map_warped[x, y] = value

    print(np.all((np.abs(skinning_map_warped.sum(axis=-1) - 1.0) < 0.01) | np.any(mesh_mask < 125, axis=2)))

    # cv2.imshow('5', ((np.abs(skinning_map_warped.sum(axis=-1) - 1.0) < 0.01) | np.any(mesh_mask < 125, axis=2)).astype(np.float32))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    uv = renderer.get_uv_coords()
    scaled_uv = (uv * np.array([[skinning_map_warped.shape[1], skinning_map_warped.shape[0]]])).astype(np.int32)
    scaled_uv[:, 1] = skinning_map_warped.shape[0] - scaled_uv[:, 1]  # reverse y axis
    # img = np.zeros((800, 800))
    # img[scaled_uv[:, 1], scaled_uv[:, 0]] += 1
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    skinning_weights = skinning_map_warped[scaled_uv[:, 1], scaled_uv[:, 0], :]
    print(f"is skinning weights sum to 1? {np.all((np.abs(skinning_weights.sum(axis=-1) - 1.0) < 0.01))}")
    problematic_vertices = ~(np.abs(skinning_weights.sum(axis=-1) - 1.0) < 0.01)
    for v in np.nonzero(problematic_vertices)[0]:
        y, x = scaled_uv[v]
        value = np.zeros(skinning_weights.shape[-1])
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                value += skinning_map_warped[i, j]
        value = value / np.sum(value)
        skinning_weights[v] = value
    # skinning_weights = skinning_weights / np.expand_dims(skinning_weights.sum(axis=-1), axis=-1)
    renderer.set_vertices(mesh['transformed_vertices'] + np.expand_dims(np.abs(skinning_weights.sum(axis=-1) - 1.0) < 0.01, axis=-1))
    render = renderer.render_normals()[0]
    cv2.imshow('render', (np.flip(render, axis=2)+1.0)/2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    np.save('skinning_weights.npy', skinning_weights)
    print(f"{skinning_map_warped.shape=}")
    print(f"{scaled_uv.shape=}")
    print(f"{skinning_weights.shape=}")

    vertices = mesh['transformed_vertices']
    for i in range(22):
        renderer.set_vertices(vertices + ((skinning_weights[:, i] > 0.2).reshape(-1, 1)*np.array([[0, 0, 1]])))
        render = renderer.render_normals()[0]
        cv2.imshow('render', (np.flip(render, axis=2)+1.0)/2)
        cv2.waitKey(200)
    cv2.destroyAllWindows()
