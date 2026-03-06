import torch
import torch.nn as nn
import timm

from typing import Optional
from torchvision import models
from transformers import AutoModel, AutoConfig


def _modify_first_conv(original_conv: nn.Conv2d, input_dim: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        input_dim,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=(original_conv.bias is not None),
    )

    with torch.no_grad():
        if input_dim == 1:
            new_conv.weight.data[:, 0:1, :, :] = original_conv.weight.data.mean(dim=1, keepdim=True)
        else:
            new_conv.weight.data[:, 0:3, :, :] = original_conv.weight.data

        if original_conv.bias is not None:
            new_conv.bias.data.copy_(original_conv.bias.data)

    return new_conv


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
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.pw_conv1(x)
        x = self.gelu(x)
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
    def __init__(self, input_dim, output_dim, size: str = None, clf_mode=True):
        super().__init__()
        C = 96

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, C, kernel_size=4, stride=4),
        )

        self.clf_mode = clf_mode

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
            nn.Linear(8 * C, output_dim),
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
        if self.clf_mode:
            return self.head(x)
        else:
            return x


class ResNetExtractor(nn.Module):
    """
    ResNet (torchvision, ImageNet pretrained).

    variant      params   top-1
    ──────────── ──────── ──────
    'resnet18'   11.7M    69.8%
    'resnet34'   21.8M    73.3%
    'resnet50'   25.6M    80.9%
    'resnet101'  44.5M    81.9%
    'resnet152'  60.2M    82.3%
    """

    _MODELS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    _WEIGHTS = {
        'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
        'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
        'resnet50': models.ResNet50_Weights.IMAGENET1K_V2,
        'resnet101': models.ResNet101_Weights.IMAGENET1K_V2,
        'resnet152': models.ResNet152_Weights.IMAGENET1K_V2,
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'resnet50', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в ResNetExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        backbone = self._MODELS[size](weights=self._WEIGHTS[size])

        self.num_features = backbone.fc.in_features

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        if input_dim != 3:
            original_conv = backbone.conv1
            self.encoder[0] = _modify_first_conv(original_conv, input_dim)

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.flatten(1)  # (B, num_features)
        if self.clf_mode:
            return self.head(x)
        return x


class MobileNetV3Extractor(nn.Module):
    """
        MobileNetV3 (torchvision) — лёгкий CNN-экстрактор признаков.

        Поддерживаемые варианты:
          variant   params   top-1   num_features
          ──────── ──────── ─────── ─────────────
          'small'   2.5M     67.7%   576
          'large'   5.5M     75.3%   960
        """

    _MODELS = {"small": models.mobilenet_v3_small, "large": models.mobilenet_v3_large}

    def __init__(self, input_dim, output_dim: Optional[int] = None, size: str = 'small', clf_mode=False):
        super(MobileNetV3Extractor, self).__init__()
        assert size in self._MODELS.keys(), (
            f"Нет такой архитектуры в MobileNetV3Extractor, просьба перепроверить size: (вами задан) {size},"
            f"поддерживаемые архитектуры: {list(self._MODELS)}")

        self.clf_mode = clf_mode

        backbone = self._MODELS[size](weights='IMAGENET1K_V1')
        original_conv = backbone.features[0][0]
        backbone.features[0][0] = _modify_first_conv(original_conv, input_dim)

        self.encoder = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.num_features = backbone.classifier[0].in_features

        if self.clf_mode:
            self.final = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.num_features, output_dim)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        if self.clf_mode:
            return self.final(x)
        else:
            return x


class ShuffleNetV2Extractor(nn.Module):
    """
    ShuffleNet V2 (torchvision) — ультралёгкий CNN-экстрактор признаков.

    Поддерживаемые варианты:
      variant   params   top-1   num_features
      ──────── ──────── ─────── ─────────────
      'x0_5'   1.4M     60.6%   1024
      'x1_0'   2.3M     69.4%   1024
      'x1_5'   3.5M     72.6%   1024
      'x2_0'   7.4M     75.0%   2048
    """

    _MODELS = {
        'x0_5': models.shufflenet_v2_x0_5,
        'x1_0': models.shufflenet_v2_x1_0,
        'x1_5': models.shufflenet_v2_x1_5,
        'x2_0': models.shufflenet_v2_x2_0,
    }

    _WEIGHTS = {
        'x0_5': models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
        'x1_0': models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1,
        'x1_5': models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1,
        'x2_0': models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1,
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'x0_5', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в ShuffleNetV2Extractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        backbone = self._MODELS[size](weights=self._WEIGHTS[size])

        self.num_features = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1],
                                     nn.AdaptiveAvgPool2d(1))

        if input_dim != 3:
            original_conv = backbone.conv1[0]
            self.encoder[0][0] = _modify_first_conv(original_conv, input_dim)

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)  # (B, num_features, 1, 1) — avgpool уже внутри
        x = x.flatten(1)  # (B, num_features)
        if self.clf_mode:
            return self.head(x)
        return x


class MobileNetV2Extractor(nn.Module):
    """
    MobileNetV2 (torchvision) — лёгкий CNN-экстрактор признаков.

      params   top-1   num_features   input
      ──────── ─────── ─────────────  ──────
      3.4M     71.9%   1280           224
    """

    def __init__(self, input_dim: int, output_dim: int, size: str = None, clf_mode: bool = False):
        super().__init__()
        self.clf_mode = clf_mode

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)

        self.num_features = backbone.classifier[1].in_features

        self.encoder = backbone.features

        if input_dim != 3:
            original_conv = backbone.features[0][0]
            self.encoder[0][0] = _modify_first_conv(original_conv, input_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        if self.clf_mode:
            return self.head(x)
        return x


class ViTExtractor(nn.Module):
    """
    Vision Transformer (torchvision) — экстрактор признаков на основе ViT.

    Поддерживаемые варианты:
      variant     patch   params   top-1   num_features   input
      ────────── ─────── ──────── ─────── ─────────────  ──────
      'base16'   16×16   86.6M    81.1%   768            224
      'base32'   32×32   88.2M    75.9%   768            224
      'large16'  16×16   307.3M   85.1%   1024           224
      'large32'  32×32   306.5M   80.2%   1024           224
      'huge14'   14×14   632.0M   88.6%   1280           518

    Входной тензор : (B, input_dim, H, W).
    'huge14' требует разрешения 518×518, остальные — 224×224.
    """

    _MODELS = {
        'base16': models.vit_b_16,
        'base32': models.vit_b_32,
        'large16': models.vit_l_16,
        'large32': models.vit_l_32,
        'huge14': models.vit_h_14,
    }

    _WEIGHTS = {
        'base16': models.ViT_B_16_Weights.IMAGENET1K_V1,
        'base32': models.ViT_B_32_Weights.IMAGENET1K_V1,
        'large16': models.ViT_L_16_Weights.IMAGENET1K_V1,
        'large32': models.ViT_L_32_Weights.IMAGENET1K_V1,
        'huge14': models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'base16', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в ViTExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        vit_model = self._MODELS[size](weights=self._WEIGHTS[size])

        if input_dim != 3:
            original_conv = vit_model.conv_proj
            vit_model.conv_proj = _modify_first_conv(original_conv, input_dim)

        self.num_features = vit_model.heads.head.in_features
        vit_model.heads = nn.Identity()
        self.encoder = vit_model

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # (B, num_features) — CLS-токен после heads=Identity
        if self.clf_mode:
            return self.head(x)
        return x


class RegNetExtractor(nn.Module):
    """
    RegNet-Y (torchvision) — экстрактор признаков на основе RegNet.

    Поддерживаемые варианты:
      variant              params    top-1   num_features   input
      ────────────────────  ────────  ──────  ─────────────  ──────
      'regnet_y_400mf'     4.3M      75.8%   440            224
      'regnet_y_800mf'     6.4M      78.8%   784            224
      'regnet_y_1_6gf'     11.2M     80.9%   888            224
      'regnet_y_3_2gf'     19.4M     82.0%   1512           224
      'regnet_y_8gf'       39.4M     82.8%   2016           224
      'regnet_y_16gf'      83.6M     86.0%   3024           224
      'regnet_y_32gf'      145.0M    86.8%   3712           224
      'regnet_y_128gf'     644.8M    88.2%   7392           384  ← SWAG pretrain

    Входной тензор : (B, input_dim, H, W).
    'regnet_y_128gf' требует разрешения 384×384.
    """

    _MODELS = {
        'regnet_y_400mf': models.regnet_y_400mf,
        'regnet_y_800mf': models.regnet_y_800mf,
        'regnet_y_1_6gf': models.regnet_y_1_6gf,
        'regnet_y_3_2gf': models.regnet_y_3_2gf,
        'regnet_y_8gf': models.regnet_y_8gf,
        'regnet_y_16gf': models.regnet_y_16gf,
        'regnet_y_32gf': models.regnet_y_32gf,
        'regnet_y_128gf': models.regnet_y_128gf,
    }

    _WEIGHTS = {
        'regnet_y_400mf': models.RegNet_Y_400MF_Weights.IMAGENET1K_V2,
        'regnet_y_800mf': models.RegNet_Y_800MF_Weights.IMAGENET1K_V2,
        'regnet_y_1_6gf': models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2,
        'regnet_y_3_2gf': models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2,
        'regnet_y_8gf': models.RegNet_Y_8GF_Weights.IMAGENET1K_V2,
        'regnet_y_16gf': models.RegNet_Y_16GF_Weights.IMAGENET1K_V2,
        'regnet_y_32gf': models.RegNet_Y_32GF_Weights.IMAGENET1K_V2,
        'regnet_y_128gf': models.RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1,
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'regnet_y_400mf', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в RegNetExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        backbone = self._MODELS[size](weights=self._WEIGHTS[size])

        self.num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        if input_dim != 3:
            original_conv = backbone.stem[0]
            backbone.stem[0] = _modify_first_conv(original_conv, input_dim)

        self.encoder = backbone

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.clf_mode:
            return self.head(x)
        return x


class EfficientNetExtractor(nn.Module):
    """
    EfficientNet B0–B7 (torchvision) — масштабируемый CNN-экстрактор признаков.

    Поддерживаемые варианты:
      variant             params   top-1   num_features   input
      ─────────────────── ──────── ─────── ─────────────  ──────
      'efficientnet-b0'   5.3M     77.7%   1280           224
      'efficientnet-b1'   7.8M     79.8%   1280           240
      'efficientnet-b2'   9.1M     80.6%   1408           288
      'efficientnet-b3'   12.2M    82.0%   1536           300
      'efficientnet-b4'   19.3M    83.4%   1792           380
      'efficientnet-b5'   30.4M    83.4%   2048           456
      'efficientnet-b6'   43.0M    84.0%   2304           528
      'efficientnet-b7'   66.3M    84.1%   2560           600

    Входной тензор : (B, input_dim, H, W).
    Рекомендуемое разрешение зависит от варианта (см. таблицу выше).
    """

    _MODELS = {
        'efficientnet-b0': models.efficientnet_b0,
        'efficientnet-b1': models.efficientnet_b1,
        'efficientnet-b2': models.efficientnet_b2,
        'efficientnet-b3': models.efficientnet_b3,
        'efficientnet-b4': models.efficientnet_b4,
        'efficientnet-b5': models.efficientnet_b5,
        'efficientnet-b6': models.efficientnet_b6,
        'efficientnet-b7': models.efficientnet_b7,
    }

    _WEIGHTS = {
        'efficientnet-b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'efficientnet-b1': models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        'efficientnet-b2': models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        'efficientnet-b3': models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        'efficientnet-b4': models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        'efficientnet-b5': models.EfficientNet_B5_Weights.IMAGENET1K_V1,
        'efficientnet-b6': models.EfficientNet_B6_Weights.IMAGENET1K_V1,
        'efficientnet-b7': models.EfficientNet_B7_Weights.IMAGENET1K_V1,
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'efficientnet-b0', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в EfficientNetExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        backbone = self._MODELS[size](weights=self._WEIGHTS[size])

        self.num_features = backbone.classifier[1].in_features

        backbone.classifier = nn.Identity()

        if input_dim != 3:
            original_conv = backbone.features[0][0]
            backbone.features[0][0] = _modify_first_conv(original_conv, input_dim)

        self.encoder = backbone

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.clf_mode:
            return self.head(x)
        return x


class DeiTExtractor(nn.Module):
    """
        DeiT (Data-efficient Image Transformers, Facebook Research).
        Загружается через torch.hub из facebookresearch/deit.

        Поддерживаемые варианты:
          variant                 params   top-1   num_features   input   distilled
          ─────────────────────── ──────── ─────── ─────────────  ──────  ─────────
          'tiny-224'              5.7M     72.2%   192            224     ❌
          'small-224'             22.1M    79.8%   384            224     ❌
          'base-224'              86.6M    81.8%   768            224     ❌
          'tiny-distilled-224'    5.9M     74.5%   192            224     ✅
          'base-distilled-224'    87.3M    83.4%   768            224     ✅
          'base-384'              86.9M    82.9%   768            384     ❌
          'base-distilled-384'    87.3M    85.2%   768            384     ✅

        Аргументы:
            input_dim  (int)  : количество входных каналов (1 — grayscale, 3 — RGB).
            output_dim (int)  : размер выхода в режиме clf_mode=True.
            size       (str)  : вариант архитектуры. По умолчанию 'tiny-224'.
            clf_mode   (bool) : True → логиты через head (B, output_dim),
                                False → CLS-токен (B, num_features).

        Входной тензор : (B, input_dim, H, W).
        'base-384' и 'base-distilled-384' требуют разрешения 384×384.
        Требует подключения к интернету при первом запуске (torch.hub).
    """

    _MODELS = {
        "tiny-224": "deit_tiny_patch16_224",
        "small-224": "deit_small_patch16_224",
        "base-224": "deit_base_patch16_224",
        "tiny-distilled-224": "deit_tiny_distilled_patch16_224",
        "base-distilled-224": "deit_base_distilled_patch16_224",
        "base-384": "deit_base_patch16_384",
        "base-distilled-384": "deit_base_distilled_patch16_384",
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'tiny-224', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в DeiTExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )

        self.clf_mode = clf_mode

        self.backbone = torch.hub.load(
            'facebookresearch/deit:main',
            self._MODELS[size],
            pretrained=True,
        )

        if input_dim != 3:
            original_conv = self.backbone.patch_embed.proj
            self.backbone.patch_embed.proj = _modify_first_conv(original_conv, input_dim)

        self.num_features = self.backbone.head.in_features

        new_head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )
        self.backbone.head = new_head

        if hasattr(self.backbone, 'head_dist'):
            self.backbone.head_dist = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clf_mode:
            out = self.backbone(x)
            if isinstance(out, tuple):
                out = out[0]
            return out
        else:
            return self.backbone.forward_features(x)[:, 0]


class DeiT3Extractor(nn.Module):
    """
        DeiT V3  — загружается через timm.
        Предобучен на ImageNet-22k, дообучен на ImageNet-1k.

        Поддерживаемые варианты:
          variant   patch   params    top-1   num_features   input
          ──────── ─────── ────────── ─────── ─────────────  ──────
          'small'  16×16   22.1M      81.4%   384            224
          'base'   16×16   86.6M      85.7%   768            224
          'large'  16×16   307.3M     87.7%   1024           224
          'huge'   14×14   632.1M     88.6%   1280           224

        Входной тензор : (B, input_dim, 224, 224).
    """

    _MODELS = {
        'small': 'deit3_small_patch16_224.fb_in22k_ft_in1k',  # 384-dim
        'base': 'deit3_base_patch16_224.fb_in22k_ft_in1k',  # 768-dim
        'large': 'deit3_large_patch16_224.fb_in22k_ft_in1k',  # 1024-dim
        'huge': 'deit3_huge_patch14_224.fb_in22k_ft_in1k',  # 1280-dim
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'small', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS.keys(), (
            f"Нет такой архитектуры в DeiT3Extractor, просьба перепроверить size: (вами задан) {size},"
            f"поддерживаемые архитектуры: {list(self._MODELS)}")
        self.clf_mode = clf_mode

        self.backbone = timm.create_model(
            self._MODELS[size],
            pretrained=True,
            num_classes=0,
            in_chans=input_dim,
        )

        self.num_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.clf_mode:
            return self.head(features)
        return features


class DINOv2Extractor(nn.Module):
    """
        DINOv2 (Meta AI) — Self-supervised ViT, предобучен на LVD-142M.
        Загружается через timm. Варианты с суффиксом '_reg' содержат
        4 регистровых токена (register tokens), что улучшает качество
        признаков на однородных областях изображения.

        Поддерживаемые варианты:
          variant      patch   params    top-1   num_features   registers   input
          ──────────── ─────── ────────── ─────── ─────────────  ─────────── ──────
          'small'      14×14   22.1M      81.1%   384            ❌          518
          'base'       14×14   86.6M      86.5%   768            ❌          518
          'large'      14×14   307.3M     87.6%   1024           ❌          518
          'giant'      14×14   1100.0M    86.5%   1536           ❌          518
          'small_reg'  14×14   22.1M      82.0%   384            ✅ reg4     518
          'base_reg'   14×14   86.6M      87.0%   768            ✅ reg4     518
          'large_reg'  14×14   307.3M     88.0%   1024           ✅ reg4     518
          'giant_reg'  14×14   1100.0M    87.2%   1536           ✅ reg4     518

        Входной тензор : (B, input_dim, 518, 518).
        ⚠️  Нативное разрешение 518×518 (patch14, 37×37 патчей).
            Передача 224×224 возможна, но снижает качество признаков.
        """

    _MODELS = {
        'small': 'vit_small_patch14_dinov2.lvd142m',
        'base': 'vit_base_patch14_dinov2.lvd142m',
        'large': 'vit_large_patch14_dinov2.lvd142m',
        'giant': 'vit_giant_patch14_dinov2.lvd142m',  # 1536-dim, 1.1B params
        'small_reg': 'vit_small_patch14_reg4_dinov2.lvd142m',
        'base_reg': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'large_reg': 'vit_large_patch14_reg4_dinov2.lvd142m',
        'giant_reg': 'vit_giant_patch14_reg4_dinov2.lvd142m',  # 1536-dim, 1.1B params
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'small_reg', clf_mode: bool = False):
        super().__init__()

        assert size in self._MODELS, (
            f"Нет такой архитектуры в DINOv2Extractor, просьба перепроверить size: (вами задан) {size},"
            f"поддерживаемые архитектуры: {list(self._MODELS)}")
        self.clf_mode = clf_mode

        self.backbone = timm.create_model(
            self._MODELS[size],
            pretrained=True,
            num_classes=0,  # → model(x) вернёт CLS (B, embed_dim)
            in_chans=input_dim,
        )

        self.num_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.clf_mode:
            return self.head(features)
        return features


class DINOv3Extractor(nn.Module):
    """
    DINOv3 (Meta AI) — Self-supervised модели, предобученные на LVD-1689M.
    Поддерживает ViT и ConvNeXt архитектуры.
    Загрузка: сначала HuggingFace, при недоступности — локальный state_dict.

    Поддерживаемые варианты:
      variant              arch      params    num_features
      ─────────────────── ───────── ────────── ─────────────
      'vits'              ViT-S/16  22.1M      384
      'vits_plus'         ViT-S/16  22.1M      384
      'vitl_text_encoder' ViT-L/16  307.3M     1024
      'vith_plus'         ViT-H/14  1100.0M    1280
      'convnext_tiny'     ConvNeXt  28.6M      768
      'convnext_small'    ConvNeXt  50.2M      768
      'convnext_base'     ConvNeXt  88.6M      1024
      'convnext_large'    ConvNeXt  197.8M     1536

    Входной тензор : (B, input_dim, H, W), рекомендуемое разрешение 224×224.
    При недоступности HuggingFace требует папку dinov3_weights/ с .pth файлами.
    AutoConfig.from_pretrained в fallback-ветке также обращается к сети.
        Для полного оффлайн-режима положите config.json рядом с .pth файлами.
    """

    _MODELS = {
        'vits': 'dinov3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'vits_plus': 'dinov3_weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        'vitl_text_encoder': 'dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'vith_plus': 'dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
        'convnext_small': 'dinov3_weights/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
        'convnext_tiny': 'dinov3_weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
        'convnext_base': 'dinov3_weights/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
        'convnext_large': 'dinov3_weights/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
    }

    _HF_MODELS = {
        'vits': 'facebook/dinov2-small',
        'vits_plus': 'facebook/dinov2-small',
        'vitl_text_encoder': 'facebook/dinov2-large',
        'vith_plus': 'facebook/dinov2-giant',
        'convnext_small': 'facebook/convnext-small-224',
        'convnext_tiny': 'facebook/convnext-tiny-224',
        'convnext_base': 'facebook/convnext-base-224',
        'convnext_large': 'facebook/convnext-large-224',
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'vits', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в DINOv3Extractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode
        self.is_convnext = 'convnext' in size  # нужно в forward
        hf_id = self._HF_MODELS[size]

        try:
            print(f"[DINOv3] Загрузка с HuggingFace: {hf_id}")
            self.backbone = AutoModel.from_pretrained(hf_id)
            print(f"[DINOv3] Успешно загружено с HuggingFace")
        except Exception as e:
            local_path = self._MODELS[size]
            print(f"[DINOv3] HuggingFace недоступен ({e})\n"
                  f"[DINOv3] Загрузка локального state_dict: {local_path}")
            config = AutoConfig.from_pretrained(hf_id)
            self.backbone = AutoModel.from_config(config)
            state_dict = torch.load(local_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f"[DINOv3] Успешно загружено локально")

        if input_dim != 3:
            if self.is_convnext:
                original_conv = self.backbone.stem[0]
                self.backbone.stem[0] = _modify_first_conv(original_conv, input_dim)
            else:
                proj = self.backbone.embeddings.patch_embeddings.projection
                self.backbone.embeddings.patch_embeddings.projection = _modify_first_conv(proj, input_dim)

        if self.is_convnext:
            self.num_features = self.backbone.config.hidden_sizes[-1]
        else:
            self.num_features = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        features = outputs.pooler_output if self.is_convnext else outputs.last_hidden_state[:, 0]
        if self.clf_mode:
            return self.head(features)
        return features


class EfficientViTExtractor(nn.Module):
    """
        EfficientViT (timm) — два семейства: MIT (B-series) и MSRA (M-series).

        MIT B-series (Multi-Scale Linear Attention):
          variant     params   top-1   num_features   input
          ────────── ──────── ─────── ─────────────  ──────
          'b0_224'   3.4M     71.9%   192            224
          'b1_224'   9.1M     79.4%   256            224
          'b1_288'   9.1M     79.9%   256            288
          'b2_224'   24.3M    82.7%   384            224
          'b2_288'   24.3M    83.0%   384            288
          'b2_384'   24.3M    83.7%   384            384
          'b3_224'   48.9M    84.2%   512            224
          'b3_288'   48.9M    84.5%   512            288
          'b3_384'   48.9M    84.9%   512            384

        MSRA M-series (Cascade Group Attention):
          variant   params   top-1   num_features   input
          ──────── ──────── ─────── ─────────────  ──────
          'm0'     2.3M     63.2%   128            224
          'm1'     2.9M     68.4%   128            224
          'm2'     4.2M     128     70.8%          224
          'm3'     6.9M     73.4%   192            224
          'm4'     8.8M     74.3%   192            224
          'm5'     12.4M    77.1%   192            224

        Входной тензор : (B, input_dim, H, W).
        Разрешение должно соответствовать суффиксу варианта (224 / 288 / 384).
        """

    _MODELS = {
        'b0_224': 'efficientvit_b0.r224_in1k',
        'b1_224': 'efficientvit_b1.r224_in1k',
        'b2_224': 'efficientvit_b2.r224_in1k',
        'b3_224': 'efficientvit_b3.r224_in1k',

        'b1_288': 'efficientvit_b1.r288_in1k',
        'b2_288': 'efficientvit_b2.r288_in1k',
        'b3_288': 'efficientvit_b3.r288_in1k',

        'b2_384': 'efficientvit_b2.r384_in1k',
        'b3_384': 'efficientvit_b3.r384_in1k',

        'm0': 'efficientvit_m0.r224_in1k',
        'm1': 'efficientvit_m1.r224_in1k',
        'm2': 'efficientvit_m2.r224_in1k',
        'm3': 'efficientvit_m3.r224_in1k',
        'm4': 'efficientvit_m4.r224_in1k',
        'm5': 'efficientvit_m5.r224_in1k',
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'b0_224', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS.keys(), (
            f"Нет такой архитектуры в EfficientViTExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        self.backbone = timm.create_model(
            self._MODELS[size],
            pretrained=True,
            num_classes=0,
            in_chans=input_dim,
        )

        with torch.no_grad():
            cfg = self.backbone.default_cfg
            h, w = cfg['input_size'][1], cfg['input_size'][2]
            dummy = torch.zeros(1, input_dim, h, w)
            out = self.backbone(dummy)
            if out.dim() == 4:
                out = out.mean([-2, -1])
            self.num_features = out.shape[1]

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean([-2, -1])
        if self.clf_mode:
            return self.head(features)
        return features


class MobileViTV2Extractor(nn.Module):
    """
        MobileViT v2 (Apple, cvnets) — лёгкий гибридный CNN+ViT экстрактор признаков.
        Загружается через timm. Варианты v2_125/150/200 предобучены на ImageNet-22k
        и дообучены на ImageNet-1k при разрешении 384×384.

        Поддерживаемые варианты:
          variant    params   top-1   num_features   input   pretrain
          ────────── ──────── ─────── ─────────────  ──────  ──────────────
          'v2_050'   1.4M     70.2%   256            256     IN-1k
          'v2_075'   2.9M     75.6%   384            256     IN-1k
          'v2_100'   4.9M     78.1%   512            256     IN-1k
          'v2_125'   7.5M     81.5%   640            384     IN-22k→IN-1k
          'v2_150'   10.6M    82.5%   768            384     IN-22k→IN-1k
          'v2_200'   18.4M    83.4%   1024           384     IN-22k→IN-1k

        Входной тензор : (B, input_dim, H, W).
        ⚠️  v2_050/075/100 — рекомендуемое разрешение 256×256.
        ⚠️  v2_125/150/200 — рекомендуемое разрешение 384×384.
        """

    _MODELS = {
        'v2_050': 'mobilevitv2_050.cvnets_in1k',
        'v2_075': 'mobilevitv2_075.cvnets_in1k',
        'v2_100': 'mobilevitv2_100.cvnets_in1k',
        'v2_125': 'mobilevitv2_125.cvnets_in22k_ft_in1k_384',
        'v2_150': 'mobilevitv2_150.cvnets_in22k_ft_in1k_384',
        'v2_200': 'mobilevitv2_200.cvnets_in22k_ft_in1k_384',
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'v2_050', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в MobileViTV2Extractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )

        self.clf_mode = clf_mode

        self.backbone = timm.create_model(
            self._MODELS[size],
            pretrained=True,
            num_classes=0,
            in_chans=3,
        )

        if input_dim != 3:
            original_conv = self.backbone.stem.conv
            self.backbone.stem.conv = _modify_first_conv(original_conv, input_dim)

        self.num_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        if self.clf_mode:
            return self.head(features)
        return features


class MobileViTExtractor(nn.Module):
    """
        MobileViT v1 (Apple, cvnets) — лёгкий гибридный CNN+ViT экстрактор признаков.
        Загружается через timm. Предобучен на ImageNet-1k.

        Поддерживаемые варианты:
          variant   params   top-1   num_features   input
          ──────── ──────── ─────── ─────────────  ──────
          'xxs'    1.3M     69.0%   320            256
          'xs'     2.3M     74.8%   384            256
          's'      5.6M     78.4%   640            256

        Входной тензор : (B, input_dim, 256, 256).
        """

    _MODELS = {
        'xxs': 'mobilevit_xxs.cvnets_in1k',
        'xs': 'mobilevit_xs.cvnets_in1k',
        's': 'mobilevit_s.cvnets_in1k',
    }

    def __init__(self, input_dim: int, output_dim: int, size: str = 'xxs', clf_mode: bool = False):
        super().__init__()
        assert size in self._MODELS, (
            f"Нет такой архитектуры в MobileViTExtractor, просьба перепроверить size: (вами задан) {size}, "
            f"поддерживаемые архитектуры: {list(self._MODELS)}"
        )
        self.clf_mode = clf_mode

        self.backbone = timm.create_model(
            self._MODELS[size],
            pretrained=True,
            num_classes=0,
            in_chans=3,
        )

        if input_dim != 3:
            original_conv = self.backbone.stem.conv
            self.backbone.stem.conv = _modify_first_conv(original_conv, input_dim)

        self.num_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        if self.clf_mode:
            return self.head(features)
        return features


class FeatureExtractor:
    def __init__(self):
        self.model_extractors = {"resnet": ResNetExtractor,
                                 "mobilenetv2": MobileNetV2Extractor,
                                 "shufflenetV2": ShuffleNetV2Extractor,
                                 "vit": ViTExtractor,
                                 "mobilenetv3": MobileNetV3Extractor,
                                 "regnet": RegNetExtractor,
                                 "efficientnet": EfficientNetExtractor,
                                 "efficientvit": EfficientViTExtractor,
                                 "deit": DeiTExtractor,
                                 "deit3": DeiT3Extractor,
                                 "mobilevit": MobileViTExtractor,
                                 "mobilevitv2": MobileViTExtractor,
                                 "convnext": ConvNext}

    def register_model(self, model_name: str, model_class: object) -> None:
        if model_name.lower() in self.model_extractors:
            raise KeyError("Model {} уже существует в классе".format(model_name))

        self.model_extractors[model_name] = model_class
        print('Модель успешно зарегистрирована')

    def get_model(self, model_name: str, size: str, input_dim: int, output_dim: int, clf_mode=False):
        if model_name.lower() not in self.model_extractors:
            raise KeyError(
                f"Model {model_name} не существует в данном классе, просьба ее реализовать и добавить через register_model")

        if not isinstance(size, str):
            raise ValueError("size must be str")

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                f"input_dim: {input_dim} или output_dim: {output_dim} не может быть отрицательным или нулевым")

        elif isinstance(input_dim, float) or isinstance(output_dim, float):
            raise ValueError(f"input_dim: {input_dim} или output_dim: {output_dim} не могут быть рациональными числами")

        model_class = self.model_extractors[model_name]
        backbone = model_class(input_dim, output_dim, size, clf_mode)
        return backbone


if __name__ == '__main__':
    x = torch.randn(1, 1, 224, 224)

    model = ConvNext(1, 3, size='aa', clf_mode=True)

    print(model(x))
