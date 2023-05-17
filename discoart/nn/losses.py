import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

class DepthLoss(nn.Module):
    def __init__(self, trim=0.2):
        super().__init__()
        self.trim = trim
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        self.transform = T.Compose(
            [
                T.Resize((384, 384)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, x, y):
        x = self.transform(x)
        y = self.transform(y)
        x = self.model(x)
        y = self.model(y)
        diff = (x - y).abs().view(-1)
        trimmed, _ = torch.sort(diff, descending=False)[:int(len(diff) * (1.0 - self.trim))]
        return trimmed.mean()
