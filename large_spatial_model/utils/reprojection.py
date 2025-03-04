import torch
import torch.nn.functional as F


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [*grid_shape, len(grid_shape)]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)


def reprojector(image, image_grid, depth, K1, K2, R, T):
    depth = depth[0, ..., None]  # noqa
    pts = torch.cat([image_grid * depth, depth], dim=-1).flatten(0, 1)
    pts = K1.inverse() @ pts.T
    # pts = E1[:3, :3].inverse() @ (pts - E1[:3, 3:4])
    # pts = E2[:3, :3] @ pts + E2[:3, 3:4]
    # pts = E1[:3, :3] @ pts + E1[:3, 3:4]
    # pts = E2[:3, :3].inverse() @ (pts - E2[:3, 3:4])
    pts = torch.from_numpy(R).to(pts).T @ pts + torch.from_numpy(T).to(pts).unsqueeze(1)
    pts = K2 @ pts
    pts = pts.T
    pts = pts / pts[:, 2:3] / torch.tensor([256, 256, 1]).to(pts.device)
    mask = torch.where((pts[:, 0] >= 0) & (pts[:, 0] <= 1) & (pts[:, 1] >= 0) &
                       (pts[:, 1] <= 1) & (pts[:, 2] >= 0), 1, 0)
    rpj_img = F.grid_sample(image[None], pts[..., :2].reshape(1, 256, 256, 2) * 2 - 1)
    return rpj_img, mask


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
        x = cmap(x / (x.max() + 1e-6))[..., 0, :3]
    x -= x.min()
    x /= x.max()
    import matplotlib.pyplot as plt
    plt.imsave(filename + '.png', x)
