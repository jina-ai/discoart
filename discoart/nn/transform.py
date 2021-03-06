import torch


def symmetry_transformation_fn(x, use_horizontal_symmetry, use_vertical_symmetry):
    if use_horizontal_symmetry:
        [n, c, h, w] = x.size()
        x = torch.concat(
            (x[:, :, :, : w // 2], torch.flip(x[:, :, :, : w // 2], [-1])), -1
        )
    if use_vertical_symmetry:
        [n, c, h, w] = x.size()
        x = torch.concat(
            (x[:, :, : h // 2, :], torch.flip(x[:, :, : h // 2, :], [-2])), -2
        )
    return x
