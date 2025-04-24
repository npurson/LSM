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
