import numpy as np


class Warp:
    def __init__(self, warp_fn):
        self.warp_fn = warp_fn

    def __call__(self, map_img):
        map_img_projected = np.zeros_like(map_img)
        for corr in self.warp_fn:
            map_img_projected[corr[0][1], corr[0][0]] = map_img[corr[1][1], corr[1][0]]
        return map_img_projected

