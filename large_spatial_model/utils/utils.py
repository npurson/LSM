import numpy as np


def save_ply(means3d, colors, opacity, scales, rotations):
    means3d = means3d.detach().cpu().numpy()
    normals = np.zeros_like(means3d)
    colors = (
        colors.detach().cpu().numpy()
        if colors.shape[-1] == 3 else np.random.rand(colors.shape[0], 3))
    opacities = opacity.detach().cpu().numpy()
    scales = np.log(scales.detach().cpu().numpy())
    rotations = rotations.detach().cpu().numpy()

    attributes = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1',
                  'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                  'rot_0', 'rot_1', 'rot_2', 'rot_3')
    dtypes = [(attr, 'f4') for attr in attributes]

    elements = np.empty(means3d.shape[0], dtype=dtypes)
    attributes = np.concatenate(
        [means3d, normals, colors, opacities, scales, rotations], axis=1)
    elements[:] = list(map(tuple, attributes))

    import time

    from plyfile import PlyData, PlyElement
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(f'{time.time() % 1e6:.2f}.ply')


color_map = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0],
    [255, 255, 255],
    [0, 128, 0],
    [128, 0, 0],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 0],
    [128, 128, 128],
    [255, 165, 0],
    [255, 215, 0],
    [0, 191, 255],
    [135, 206, 250],
    [199, 21, 133],
    [255, 20, 147],
    [255, 105, 180],
    [255, 182, 193],
    [255, 228, 225],
    [160, 82, 45],
    [139, 69, 19],
    [210, 105, 30],
    [244, 164, 96],
    [222, 184, 135],
    [188, 143, 143],
    [112, 128, 144],
    [119, 136, 153],
    [47, 79, 79],
    [105, 105, 105],
    [169, 169, 169],
    [0, 255, 127],
    [46, 139, 87],
    [60, 179, 113],
    [32, 178, 170],
    [64, 224, 208],
    [72, 209, 204],
    [0, 139, 139],
    [0, 206, 209],
    [127, 255, 212],
    [175, 238, 238],
    [176, 224, 230],
    [95, 158, 160],
    [100, 149, 237],
    [30, 144, 255],
    [0, 0, 255],
    [65, 105, 225],
    [138, 43, 226],
    [75, 0, 130],
    [72, 61, 139],
    [106, 90, 205],
    [123, 104, 238],
    [147, 112, 219],
    [139, 0, 139],
    [148, 0, 211],
    [186, 85, 211],
    [153, 50, 204],
    [221, 160, 221],
    [238, 130, 238],
    [255, 0, 255],
    [218, 112, 214],
    [199, 21, 133],
    [219, 112, 147],
    [255, 160, 122],
    [250, 128, 114],
    [233, 150, 122],
    [240, 128, 128],
    [205, 92, 92],
]


def visualize(x, filename):
     if isinstance(x, list):
         x = torch.stack(x)  # B, 3/1, H, W
     if x.dim() == 3:
         x = x.unsqueeze(1)
     assert x.size(1) in (1, 3)
     x = x.permute(2, 0, 3, 1).flatten(1, 2).detach().cpu().numpy()
     if x.shape[-1] == 1:
         from matplotlib import cm
         cmap = cm.get_cmap('Spectral_r')
         x -= x.min()
         x /= x.max()
         x = cmap(x)[..., 0, :3]
     x -= x.min()
     x /= x.max()
     import matplotlib.pyplot as plt
     plt.imsave(filename + '.png', x)
