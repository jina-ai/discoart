import torch
import torchvision.transforms as T

inv_normalize = T.Normalize(
    mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
    std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711],
)


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
