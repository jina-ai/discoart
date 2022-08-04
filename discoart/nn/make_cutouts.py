import torch
from resize_right import resize
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF


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
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.RandomGrayscale(p=0.1),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def forward(self, input):
        return torch.cat([self.augment(c) for c in self._cut_generator(input)])

    def _cut_generator(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
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

        cutout = resize(pad_input, out_shape=output_shape)
        for j in range(self.Overview):
            if j == 1:
                cutouts.append(gray(cutout))
            elif j == 2:
                cutouts.append(TF.hflip(cutout))
            elif j == 3:
                cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutouts.append(cutout)

        for i in range(self.InnerCrop):
            size = int(
                torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            if i <= int(self.IC_Grey_P * self.InnerCrop):
                cutout = gray(cutout)
            cutouts.append(resize(cutout, out_shape=output_shape))
        return cutouts
