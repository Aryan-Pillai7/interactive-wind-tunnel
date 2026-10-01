"""Small 2D U-Net, about 1-2M parameters. (Owner: Abhay; needs torch)

Input: normalised [N, 3, 64, 128] (sdf, mask, re_norm).
Output: normalised [N, 3, 64, 128] (u, v, p).
Normalisation, denormalisation and masking are added around this network
at export time so the ONNX graph takes raw inputs (see ABHAY.md).
"""

import torch
from torch import nn

from windtunnel import contract as C

DEFAULT_CONFIG = {"in_channels": C.N_IN, "out_channels": C.N_OUT,
                  "base_channels": 16, "depth": 4, "groups": 8}


class DoubleConv(nn.Module):
    """(3x3 conv -> GroupNorm -> GELU) x 2."""

    def __init__(self, cin, cout, groups):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.GELU())

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net with `depth` 2x downsamplings; channels double at each level.

    64 x 128 halves cleanly up to depth 4 (bottleneck 4 x 8).
    """

    def __init__(self, in_channels=C.N_IN, out_channels=C.N_OUT, base_channels=16, depth=4,
                 groups=8):
        super().__init__()
        assert C.NY % 2**depth == 0 and C.NX % 2**depth == 0, "grid must halve cleanly"
        self.config = dict(in_channels=in_channels, out_channels=out_channels,
                           base_channels=base_channels, depth=depth, groups=groups)
        chans = [base_channels * 2**k for k in range(depth + 1)]
        self.stem = DoubleConv(in_channels, chans[0], groups)
        self.down = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(chans[k], chans[k + 1], groups))
            for k in range(depth))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(chans[k + 1], chans[k], 2, stride=2) for k in reversed(range(depth)))
        self.dec = nn.ModuleList(
            DoubleConv(2 * chans[k], chans[k], groups) for k in reversed(range(depth)))
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []
        for down in self.down:
            skips.append(x)
            x = down(x)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def build_model(config=None):
    return UNet(**{**DEFAULT_CONFIG, **(config or {})})


def param_count(model):
    return sum(p.numel() for p in model.parameters())


class Exported(nn.Module):
    """Raw inputs -> masked physical fields: the exact ONNX contract (ABHAY.md section 4)."""

    def __init__(self, net, in_mean, in_std, out_mean, out_std):
        super().__init__()
        self.net = net
        for k, v in dict(in_mean=in_mean, in_std=in_std, out_mean=out_mean, out_std=out_std).items():
            self.register_buffer(k, torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, inputs):
        mask = inputs[:, C.IN_MASK:C.IN_MASK + 1]
        y = self.net((inputs - self.in_mean) / self.in_std)
        y = y * self.out_std + self.out_mean
        return y * (1.0 - mask)


if __name__ == "__main__":
    m = build_model()
    print(m.config, f"params={param_count(m):,}")
