import torch
import torch.nn as nn


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels):
        super(DepthwiseConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)

    def forward(self, x):
        x = self.depthwise(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.size()
            mask = torch.rand(B, 1, 1, 1, device=x.device) > self.drop_prob
        else:
            B, T, C, H, W = x.size()
            mask = torch.rand(B, 1, 1, 1, 1, device=x.device) > self.drop_prob

        if not self.training:
            return x
        else:
            return x * mask.float() / (1 - self.drop_prob)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        super(LayerScale, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.gamma.view(1, -1, 1, 1)


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNextBlock, self).__init__()

        self.depthwise = DepthwiseConv2d(in_channels)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.pw_conv1 = nn.Conv2d(in_channels, 4 * out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv2d(4 * in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer_scale = LayerScale(out_channels)
        self.drop_path = DropPath(drop_prob=0.3)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)

        # Depthwise
        x = self.depthwise(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)

        x = self.layer_scale(x)
        x = self.drop_path(x)

        return shortcut + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        self.layer_norm = nn.LayerNorm(in_channels)
        self.pw_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pw_conv2(x)
        return x


class ConvNext(nn.Module):
    def __init__(self, in_channels=3, base_channels=96, num_classes=1000):
        super().__init__()
        C = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=4, stride=4),
        )

        self.layer_norm = nn.LayerNorm(C)

        self.stage1 = nn.Sequential(*[ConvNextBlock(C, C) for _ in range(3)])
        self.down1 = DownSample(C, 2 * C)

        self.stage2 = nn.Sequential(*[ConvNextBlock(2 * C, 2 * C) for _ in range(3)])
        self.down2 = DownSample(2 * C, 4 * C)

        self.stage3 = nn.Sequential(*[ConvNextBlock(4 * C, 4 * C) for _ in range(9)])
        self.down3 = DownSample(4 * C, 8 * C)

        self.stage4 = nn.Sequential(*[ConvNextBlock(8 * C, 8 * C) for _ in range(3)])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8 * C, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.down3(self.stage3(x))
        x = self.stage4(x)
        return self.head(x)
