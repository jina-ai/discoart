import torch
from resize_right import resize
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class Resize(nn.Module):
    def __init__(self, cut_size):
        super().__init__()
        self.cut_size = cut_size

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        output_shape = [input.shape[0], 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
        )

        return resize(pad_input, out_shape=output_shape)


class MakeCutouts(nn.Module):
    def __init__(
        self,
        cut_size,
        Overview=4,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augment = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def forward(self, input):
        return torch.cat([self.augment(c) for c in self._cut_generator(input)])

    def _cut_generator(self, cutout):
        gray = T.Grayscale(3)

        for j in range(self.Overview):
            if j == 1:
                yield gray(cutout)
            elif j == 2:
                yield TF.hflip(cutout)
            elif j == 3:
                yield gray(TF.hflip(cutout))
            else:
                yield cutout
