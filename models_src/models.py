class ResNetExtractor:
    """
    ResNet (timm, ImageNet-1k, AugReg).

    variant      timm tag               params   top-1
    ──────────── ────────────────────── ──────── ──────
    'resnet18'   resnet18.a1_in1k       11.7M    71.5%
    'resnet34'   resnet34.a1_in1k       21.8M    75.1%
    'resnet50'   resnet50.a1_in1k       25.6M    80.4%
    'resnet101'  resnet101.a1_in1k      44.5M    82.3%
    'resnet152'  resnet152.a1_in1k      60.2M    82.8%
    """
    _MODELS = {
        'resnet18': 'resnet18.a1_in1k',
        'resnet34': 'resnet34.a1_in1k',
        'resnet50': 'resnet50.a1_in1k',
        'resnet101': 'resnet101.a1_in1k',
        'resnet152': 'resnet152.a1_in1k',
    }


class MobileNetV2Extractor:
    """
    MobileNetV2 (timm).

    variant   timm tag                  params   top-1
    ──────── ─────────────────────────  ──────── ──────
    'v2'     mobilenetv2_100.ra_in1k    3.4M     72.9%
    """
    _MODELS = {'v2': 'mobilenetv2_100.ra_in1k'}


class ShuffleNetV2Extractor:
    """
    ShuffleNet V2 (timm).

    variant   timm tag               params   top-1
    ──────── ────────────────────── ──────── ──────
    'x0_5'   shufflenet_v2_x0_5     1.4M     60.6%
    'x1_0'   shufflenet_v2_x1_0     2.3M     69.4%
    'x1_5'   shufflenet_v2_x1_5     3.5M     72.6%
    'x2_0'   shufflenet_v2_x2_0     7.4M     75.0%
    """
    _MODELS = {
        'x0_5': 'shufflenet_v2_x0_5',
        'x1_0': 'shufflenet_v2_x1_0',
        'x1_5': 'shufflenet_v2_x1_5',
        'x2_0': 'shufflenet_v2_x2_0',
    }


class EfficientNetExtractor:
    """
    EfficientNet B0–B7 (timm).

    variant              timm tag                         params   top-1
    ──────────────────── ──────────────────────────────── ──────── ──────
    'efficientnet-b0'    efficientnet_b0.ra_in1k           5.3M     77.7%
    'efficientnet-b1'    efficientnet_b1.ft_in1k            7.8M     79.2%
    'efficientnet-b2'    efficientnet_b2.ra_in1k            9.1M     80.6%
    'efficientnet-b3'    efficientnet_b3.ra2_in1k          12.2M    82.0%
    'efficientnet-b4'    efficientnet_b4.ra2_in1k          19.3M    83.8%
    'efficientnet-b5'    efficientnet_b5.sw_in12k_ft_in1k  30.4M    84.5%
    'efficientnet-b6'    efficientnet_b6.ra_in1k           43.0M    84.0%
    'efficientnet-b7'    efficientnet_b7.ra_in1k           66.3M    84.1%
    """
    _MODELS = {
        'efficientnet-b0': 'efficientnet_b0.ra_in1k',
        'efficientnet-b1': 'efficientnet_b1.ft_in1k',
        'efficientnet-b2': 'efficientnet_b2.ra_in1k',
        'efficientnet-b3': 'efficientnet_b3.ra2_in1k',
        'efficientnet-b4': 'efficientnet_b4.ra2_in1k',
        'efficientnet-b5': 'efficientnet_b5.sw_in12k_ft_in1k',
        'efficientnet-b6': 'efficientnet_b6.ra_in1k',
        'efficientnet-b7': 'efficientnet_b7.ra_in1k',
    }


class RegNetExtractor:
    """
    RegNet-Y (timm).

    variant              timm tag                params    top-1
    ──────────────────── ─────────────────────── ────────  ──────
    'regnet_y_400mf'     regnety_004.tv2_in1k     4.3M     75.8%
    'regnet_y_800mf'     regnety_008.tv2_in1k     6.4M     78.8%
    'regnet_y_1_6gf'     regnety_016.tv2_in1k    11.2M     80.9%
    'regnet_y_3_2gf'     regnety_032.tv2_in1k    19.4M     82.0%
    'regnet_y_8gf'       regnety_080.tv2_in1k    39.4M     82.8%
    'regnet_y_16gf'      regnety_160.tv2_in1k    83.6M     86.0%
    'regnet_y_32gf'      regnety_320.tv2_in1k   145.0M     86.8%
    """
    _MODELS = {
        'regnet_y_400mf': 'regnety_004.tv2_in1k',
        'regnet_y_800mf': 'regnety_008.tv2_in1k',
        'regnet_y_1_6gf': 'regnety_016.tv2_in1k',
        'regnet_y_3_2gf': 'regnety_032.tv2_in1k',
        'regnet_y_8gf': 'regnety_080.tv2_in1k',
        'regnet_y_16gf': 'regnety_160.tv2_in1k',
        'regnet_y_32gf': 'regnety_320.tv2_in1k',
    }


class ViTExtractor:
    """
    Vision Transformer (timm, AugReg/IN-21k → IN-1k).

    variant     timm tag                                          params   top-1
    ────────── ────────────────────────────────────────────────── ──────── ──────
    'base16'   vit_base_patch16_224.augreg_in21k_ft_in1k          86.6M    85.5%
    'base32'   vit_base_patch32_224.augreg_in21k_ft_in1k          88.2M    80.7%
    'large16'  vit_large_patch16_224.augreg_in21k_ft_in1k        307.3M    86.6%
    'large32'  vit_large_patch32_224.augreg_in21k_ft_in1k        306.5M    81.1%
    'huge14'   vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k   632.0M    88.6%
    """
    _MODELS = {
        'base16': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
        'base32': 'vit_base_patch32_224.augreg_in21k_ft_in1k',
        'large16': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
        'large32': 'vit_large_patch32_224.augreg_in21k_ft_in1k',
        'huge14': 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',
    }


class DeiTExtractor:
    """
    DeiT v1 (timm). Заменяет устаревший torch.hub.load('facebookresearch/deit').

    variant                timm tag                                   params   top-1
    ─────────────────────  ──────────────────────────────────────── ──────── ──────
    'tiny-224'             deit_tiny_patch16_224.fb_in1k              5.7M     72.2%
    'small-224'            deit_small_patch16_224.fb_in1k            22.1M     79.8%
    'base-224'             deit_base_patch16_224.fb_in1k             86.6M     81.8%
    'tiny-distilled-224'   deit_tiny_distilled_patch16_224.fb_in1k   5.9M     74.5%
    'base-distilled-224'   deit_base_distilled_patch16_224.fb_in1k  87.3M     83.4%
    'base-384'             deit_base_patch16_384.fb_in1k             86.9M     82.9%
    'base-distilled-384'   deit_base_distilled_patch16_384.fb_in1k  87.3M     85.2%
    """
    _MODELS = {
        'tiny-224': 'deit_tiny_patch16_224.fb_in1k',
        'small-224': 'deit_small_patch16_224.fb_in1k',
        'base-224': 'deit_base_patch16_224.fb_in1k',
        'tiny-distilled-224': 'deit_tiny_distilled_patch16_224.fb_in1k',
        'base-distilled-224': 'deit_base_distilled_patch16_224.fb_in1k',
        'base-384': 'deit_base_patch16_384.fb_in1k',
        'base-distilled-384': 'deit_base_distilled_patch16_384.fb_in1k',
    }


class DeiT3Extractor:
    """
    DeiT III (timm, IN-22k → IN-1k fine-tune).

    variant   timm tag                                      params    top-1
    ──────── ────────────────────────────────────────────── ────────  ──────
    'small'  deit3_small_patch16_224.fb_in22k_ft_in1k       22.1M    81.4%
    'base'   deit3_base_patch16_224.fb_in22k_ft_in1k        86.6M    85.7%
    'large'  deit3_large_patch16_224.fb_in22k_ft_in1k      307.3M    87.7%
    'huge'   deit3_huge_patch14_224.fb_in22k_ft_in1k       632.1M    88.6%
    """
    _MODELS = {
        'small': 'deit3_small_patch16_224.fb_in22k_ft_in1k',
        'base': 'deit3_base_patch16_224.fb_in22k_ft_in1k',
        'large': 'deit3_large_patch16_224.fb_in22k_ft_in1k',
        'huge': 'deit3_huge_patch14_224.fb_in22k_ft_in1k',
    }


class DINOv2Extractor:
    """
    DINOv2 (timm, self-supervised LVD-142M).
    Варианты '_reg' содержат 4 register-токена → лучше на однородных областях.

    variant      timm tag                              params     top-1   input
    ──────────── ───────────────────────────────────── ─────────  ──────  ──────
    'small'      vit_small_patch14_dinov2.lvd142m       22.1M     81.1%   518
    'base'       vit_base_patch14_dinov2.lvd142m        86.6M     86.5%   518
    'large'      vit_large_patch14_dinov2.lvd142m      307.3M     87.6%   518
    'giant'      vit_giant_patch14_dinov2.lvd142m     1100.0M     86.5%   518
    'small_reg'  vit_small_patch14_reg4_dinov2.lvd142m  22.1M    82.0%   518
    'base_reg'   vit_base_patch14_reg4_dinov2.lvd142m   86.6M    87.0%   518
    'large_reg'  vit_large_patch14_reg4_dinov2.lvd142m 307.3M    88.0%   518
    'giant_reg'  vit_giant_patch14_reg4_dinov2.lvd142m 1100.0M   87.2%   518
    """
    _MODELS = {
        'small': 'vit_small_patch14_dinov2.lvd142m',
        'base': 'vit_base_patch14_dinov2.lvd142m',
        'large': 'vit_large_patch14_dinov2.lvd142m',
        'giant': 'vit_giant_patch14_dinov2.lvd142m',
        'small_reg': 'vit_small_patch14_reg4_dinov2.lvd142m',
        'base_reg': 'vit_base_patch14_reg4_dinov2.lvd142m',
        'large_reg': 'vit_large_patch14_reg4_dinov2.lvd142m',
        'giant_reg': 'vit_giant_patch14_reg4_dinov2.lvd142m',
    }


class EfficientViTExtractor:
    """
    EfficientViT MIT B-series и MSRA M-series (timm).

    MIT B-series (Multi-Scale Linear Attention):
      'b0_224', 'b1_224', 'b1_288', 'b2_224', 'b2_288', 'b2_384',
      'b3_224', 'b3_288', 'b3_384'

    MSRA M-series (Cascade Group Attention):
      'm0' … 'm5'
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


class MobileViTExtractor:
    """
    MobileViT v1 (timm, cvnets, IN-1k).

    variant   timm tag                params   top-1   input
    ──────── ──────────────────────── ──────── ─────── ──────
    'xxs'    mobilevit_xxs.cvnets_in1k  1.3M   69.0%   256
    'xs'     mobilevit_xs.cvnets_in1k   2.3M   74.8%   256
    's'      mobilevit_s.cvnets_in1k    5.6M   78.4%   256
    """
    _MODELS = {
        'xxs': 'mobilevit_xxs.cvnets_in1k',
        'xs': 'mobilevit_xs.cvnets_in1k',
        's': 'mobilevit_s.cvnets_in1k',
    }


class MobileViTV2Extractor:
    """
    MobileViT v2 (timm, cvnets).

    variant    timm tag                                    params   top-1   input
    ────────── ─────────────────────────────────────────── ──────── ─────── ──────
    'v2_050'   mobilevitv2_050.cvnets_in1k                  1.4M    70.2%   256
    'v2_075'   mobilevitv2_075.cvnets_in1k                  2.9M    75.6%   256
    'v2_100'   mobilevitv2_100.cvnets_in1k                  4.9M    78.1%   256
    'v2_125'   mobilevitv2_125.cvnets_in22k_ft_in1k_384     7.5M    81.5%   384
    'v2_150'   mobilevitv2_150.cvnets_in22k_ft_in1k_384    10.6M    82.5%   384
    'v2_200'   mobilevitv2_200.cvnets_in22k_ft_in1k_384    18.4M    83.4%   384
    """
    _MODELS = {
        'v2_050': 'mobilevitv2_050.cvnets_in1k',
        'v2_075': 'mobilevitv2_075.cvnets_in1k',
        'v2_100': 'mobilevitv2_100.cvnets_in1k',
        'v2_125': 'mobilevitv2_125.cvnets_in22k_ft_in1k_384',
        'v2_150': 'mobilevitv2_150.cvnets_in22k_ft_in1k_384',
        'v2_200': 'mobilevitv2_200.cvnets_in22k_ft_in1k_384',
    }


class ConvNextExtractor:
    """
    ConvNeXt v1 (timm, IN-1k / IN-22k→IN-1k).

    variant    timm tag                                       params   top-1
    ────────── ──────────────────────────────────────────── ──────── ──────
    'tiny'     convnext_tiny.a1h_in1k                        28.6M    82.1%
    'small'    convnext_small.in12k_ft_in1k                  50.2M    84.6%
    'base'     convnext_base.clip_laion2b_augreg_ft_in1k     88.6M    86.2%
    'large'    convnext_large.fb_in22k_ft_in1k              197.8M    86.6%
    'xlarge'   convnext_xlarge.fb_in22k_ft_in1k             350.2M    87.0%
    """
    _MODELS = {
        'tiny': 'convnext_tiny.a1h_in1k',
        'small': 'convnext_small.in12k_ft_in1k',
        'base': 'convnext_base.clip_laion2b_augreg_ft_in1k',
        'large': 'convnext_large.fb_in22k_ft_in1k',
        'xlarge': 'convnext_xlarge.fb_in22k_ft_in1k',
    }


class MobileNetV3Extractor:
    """
    MobileNetV3 (timm).

    variant       timm tag                                        params   top-1   input
    ────────────  ──────────────────────────────────────────────  ──────── ──────  ──────
    'small'       mobilenetv3_small_100.lamb_in1k                  2.5M    67.7%   224
    'large'       mobilenetv3_large_100.ra_in1k                    5.5M    74.0%   224
    'large_ra4'   mobilenetv3_large_100.ra4_e3600_r224_in1k        5.5M    76.3%   224  ← лучшие веса
    'large_150d'  mobilenetv3_large_150d.ra4_e3600_r256_in1k       7.5M    80.9%   256  ← wider variant
    """
    _MODELS = {
        'small': 'mobilenetv3_small_100.lamb_in1k',
        'large': 'mobilenetv3_large_100.ra_in1k',
        'large_ra4': 'mobilenetv3_large_100.ra4_e3600_r224_in1k',
        'large_150d': 'mobilenetv3_large_150d.ra4_e3600_r256_in1k',
    }


class MobileNetV4Extractor:
    """
    MobileNetV4 (timm, Google, 2024). Три серии:
      conv_*   — pure CNN, runtime-optimal на DSP/CPU/GPU
      hybrid_* — CNN + Mobile MQA attention, лучший accuracy/latency
      conv_aa_* — conv_large с anti-aliasing

    variant                   timm tag                                          params   top-1   input
    ──────────────────────── ───────────────────────────────────────────────── ──────── ──────  ──────
    'conv_small'              mobilenetv4_conv_small.e2400_r224_in1k             3.8M    74.6%   256
    'conv_medium'             mobilenetv4_conv_medium.e500_r256_in1k             9.7M    80.9%   320
    'conv_medium_in12k'       mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k   9.7M    82.4%   384  ← IN-12k pretrain
    'conv_large'              mobilenetv4_conv_large.e600_r384_in1k             32.6M    83.4%   448
    'conv_aa_large'           mobilenetv4_conv_aa_large.e600_r384_in1k          32.6M    83.8%   480  ← anti-aliased
    'conv_aa_large_in12k'     mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k 32.6M   84.8%   480  ← best conv
    'hybrid_medium'           mobilenetv4_hybrid_medium.e500_r224_in1k          11.1M    81.3%   256
    'hybrid_medium_in12k'     mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k 11.1M   83.0%   320  ← IN-12k pretrain
    'hybrid_large'            mobilenetv4_hybrid_large.e600_r384_in1k           37.8M    84.3%   448
    'hybrid_large_in12k'      mobilenetv4_hybrid_large.ix_e600_r384_in1k        37.8M    84.4%   448  ← best hybrid
    """
    _MODELS = {
        'conv_small': 'mobilenetv4_conv_small.e2400_r224_in1k',
        'conv_medium': 'mobilenetv4_conv_medium.e500_r256_in1k',
        'conv_medium_in12k': 'mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k',
        'conv_large': 'mobilenetv4_conv_large.e600_r384_in1k',
        'conv_aa_large': 'mobilenetv4_conv_aa_large.e600_r384_in1k',
        'conv_aa_large_in12k': 'mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k',
        'hybrid_medium': 'mobilenetv4_hybrid_medium.e500_r224_in1k',
        'hybrid_medium_in12k': 'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k',
        'hybrid_large': 'mobilenetv4_hybrid_large.e600_r384_in1k',
        'hybrid_large_in12k': 'mobilenetv4_hybrid_large.ix_e600_r384_in1k',
    }


class SwinTransformerExtractor:
    """
    Swin Transformer v1 (timm, IN-1k / IN-22k→IN-1k).
    Shifted-window self-attention, иерархический backbone для детекции/сегментации.

    variant    timm tag                                            params   top-1
    ────────── ─────────────────────────────────────────────────── ──────── ──────
    'tiny'     swin_tiny_patch4_window7_224.ms_in1k                28.3M    81.2%
    'small'    swin_small_patch4_window7_224.ms_in1k               49.6M    83.2%
    'base'     swin_base_patch4_window7_224.ms_in1k                87.8M    83.5%
    'base_22k' swin_base_patch4_window7_224.ms_in22k_ft_in1k       87.8M    85.3%
    'large_22k' swin_large_patch4_window7_224.ms_in22k_ft_in1k    196.5M    86.3%
    """
    _MODELS = {
        'tiny': 'swin_tiny_patch4_window7_224.ms_in1k',
        'small': 'swin_small_patch4_window7_224.ms_in1k',
        'base': 'swin_base_patch4_window7_224.ms_in1k',
        'base_22k': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
        'large_22k': 'swin_large_patch4_window7_224.ms_in22k_ft_in1k',
    }


class SwinV2Extractor:
    """
    Swin Transformer v2 (timm). Log-spaced coords → лучший transfer на другие разрешения.

    variant    timm tag                                                params   top-1   input
    ────────── ────────────────────────────────────────────────────── ──────── ──────  ──────
    'tiny'     swinv2_tiny_window8_256.ms_in1k                        28.4M    82.8%   256
    'small'    swinv2_small_window8_256.ms_in1k                       49.7M    84.1%   256
    'base'     swinv2_base_window8_256.ms_in1k                        87.9M    84.6%   256
    'large_22k' swinv2_large_window12to16_192to256.ms_in22k_ft_in1k  196.7M   87.0%   256
    """
    _MODELS = {
        'tiny': 'swinv2_tiny_window8_256.ms_in1k',
        'small': 'swinv2_small_window8_256.ms_in1k',
        'base': 'swinv2_base_window8_256.ms_in1k',
        'large_22k': 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k',
    }


class MaxViTExtractor:
    """
    MaxViT / Multi-Axis Vision Transformer (timm, Google, 2022).
    Чередующиеся window + grid attention → глобальный receptive field при O(n) сложности.

    variant   timm tag                          params   top-1   input
    ──────── ─────────────────────────────────── ──────── ──────  ──────
    'tiny'   maxvit_tiny_tf_224.in1k             30.9M    83.6%   224
    'small'  maxvit_small_tf_224.in1k            68.9M    84.5%   224
    'base'   maxvit_base_tf_224.in1k            119.9M    84.9%   224
    'large'  maxvit_large_tf_224.in1k           211.8M    85.2%   224
    'base_384'  maxvit_base_tf_384.in21k_ft_in1k  119.9M  86.6%  384
    'large_384' maxvit_large_tf_384.in21k_ft_in1k 211.8M  87.1%  384
    """
    _MODELS = {
        'tiny': 'maxvit_tiny_tf_224.in1k',
        'small': 'maxvit_small_tf_224.in1k',
        'base': 'maxvit_base_tf_224.in1k',
        'large': 'maxvit_large_tf_224.in1k',
        'base_384': 'maxvit_base_tf_384.in21k_ft_in1k',
        'large_384': 'maxvit_large_tf_384.in21k_ft_in1k',
    }


class ConvNeXtV2Extractor:
    """
    ConvNeXt V2 (timm, FCMAE + IN-22k→IN-1k).
    Fully Convolutional Masked Autoencoder pretraining + GRN.

    variant   timm tag                                         params   top-1
    ──────── ──────────────────────────────────────────────── ──────── ──────
    'atto'   convnextv2_atto.fcmae_ft_in1k                     3.7M    76.7%
    'femto'  convnextv2_femto.fcmae_ft_in1k                    5.2M    78.5%
    'pico'   convnextv2_pico.fcmae_ft_in1k                     9.1M    80.3%
    'nano'   convnextv2_nano.fcmae_ft_in22k_in1k               15.6M   82.1%
    'tiny'   convnextv2_tiny.fcmae_ft_in22k_in1k               28.6M   83.9%
    'small'  convnextv2_small.fcmae_ft_in22k_in1k              50.3M   85.8%
    'base'   convnextv2_base.fcmae_ft_in22k_in1k               88.7M   86.8%
    'large'  convnextv2_large.fcmae_ft_in22k_in1k             197.9M   87.3%
    'huge'   convnextv2_huge.fcmae_ft_in22k_in1k              660.3M   88.7%
    """
    _MODELS = {
        'atto': 'convnextv2_atto.fcmae_ft_in1k',
        'femto': 'convnextv2_femto.fcmae_ft_in1k',
        'pico': 'convnextv2_pico.fcmae_ft_in1k',
        'nano': 'convnextv2_nano.fcmae_ft_in22k_in1k',
        'tiny': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
        'small': 'convnextv2_small.fcmae_ft_in22k_in1k',
        'base': 'convnextv2_base.fcmae_ft_in22k_in1k',
        'large': 'convnextv2_large.fcmae_ft_in22k_in1k',
        'huge': 'convnextv2_huge.fcmae_ft_in22k_in1k',
    }


class EfficientNetV2Extractor:
    """
    EfficientNet V2 (timm). Fused-MBConv в ранних стадиях → быстрее B-серии.

    variant   timm tag                                           params   top-1
    ──────── ─────────────────────────────────────────────────── ──────── ──────
    's'      efficientnetv2_s.in1k                               21.5M    83.9%
    'm'      efficientnetv2_m.in1k                               54.1M    85.1%
    'l'      efficientnetv2_l.in1k                              118.5M    85.8%
    'xl_22k' tf_efficientnetv2_xl.in21k_ft_in1k                208.1M    87.3%
    's_22k'  tf_efficientnetv2_s.in21k_ft_in1k                  21.5M    84.9%
    'm_22k'  tf_efficientnetv2_m.in21k_ft_in1k                  54.1M    86.2%
    'l_22k'  tf_efficientnetv2_l.in21k_ft_in1k                 118.5M    86.9%
    """
    _MODELS = {
        's': 'efficientnetv2_s.in1k',
        'm': 'efficientnetv2_m.in1k',
        'l': 'efficientnetv2_l.in1k',
        'xl_22k': 'tf_efficientnetv2_xl.in21k_ft_in1k',
        's_22k': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'm_22k': 'tf_efficientnetv2_m.in21k_ft_in1k',
        'l_22k': 'tf_efficientnetv2_l.in21k_ft_in1k',
    }


class RepViTExtractor:
    """
    RepViT (timm, Microsoft, 2023). Re-parameterized ViT-style token mixer
    в lightweight CNN-структуре, SOTA на мобильных устройствах.

    variant   timm tag                           params   top-1   input
    ──────── ─────────────────────────────────── ──────── ──────  ──────
    'm1'     repvit_m1.dist_in1k                  5.1M    78.6%   224
    'm1_1'   repvit_m1_1.dist_in1k               6.8M    80.0%   224
    'm1_5'   repvit_m1_5.dist_in1k               9.6M    81.7%   224
    'm2'     repvit_m2.dist_in1k                11.8M    82.5%   224
    'm2_3'   repvit_m2_3.dist_in1k              22.9M    83.7%   224
    'm3'     repvit_m3.dist_in1k                10.7M    83.2%   224  ← wider
    """
    _MODELS = {
        'm1': 'repvit_m1.dist_in1k',
        'm1_1': 'repvit_m1_1.dist_in1k',
        'm1_5': 'repvit_m1_5.dist_in1k',
        'm2': 'repvit_m2.dist_in1k',
        'm2_3': 'repvit_m2_3.dist_in1k',
        'm3': 'repvit_m3.dist_in1k',
    }


class FastViTExtractor:
    """
    FastViT (timm, Apple, 2023). RepMixer token mixing → 3-6x быстрее ViT.

    variant   timm tag                        params   top-1   input
    ──────── ────────────────────────────── ──────── ──────  ──────
    't8'     fastvit_t8.apple_in1k            4.0M    76.2%   256
    't12'    fastvit_t12.apple_in1k           7.1M    79.1%   256
    's12'    fastvit_s12.apple_in1k          10.9M    79.9%   256
    'sa12'   fastvit_sa12.apple_in1k         11.6M    80.9%   256  ← +self-attn
    'sa24'   fastvit_sa24.apple_in1k         21.5M    82.6%   256
    'sa36'   fastvit_sa36.apple_in1k         31.9M    83.6%   256
    'ma36'   fastvit_ma36.apple_in1k         44.4M    83.9%   256  ← MA = medium attention
    """
    _MODELS = {
        't8': 'fastvit_t8.apple_in1k',
        't12': 'fastvit_t12.apple_in1k',
        's12': 'fastvit_s12.apple_in1k',
        'sa12': 'fastvit_sa12.apple_in1k',
        'sa24': 'fastvit_sa24.apple_in1k',
        'sa36': 'fastvit_sa36.apple_in1k',
        'ma36': 'fastvit_ma36.apple_in1k',
    }


class GhostNetExtractor:
    """
    GhostNet V1/V2 (timm). Ghost-модули: cheap linear ops вместо полных Conv.

    variant   timm tag                       params   top-1   input
    ──────── ────────────────────────────── ──────── ──────  ──────
    'v1_100' ghostnet_100.in1k               5.2M    73.9%   224
    'v1_130' ghostnet_130.in1k               7.4M    75.8%   224
    'v2_100' ghostnetv2_100.in1k             6.1M    74.4%   224
    'v2_160' ghostnetv2_160.in1k            12.4M    77.3%   224
    """
    _MODELS = {
        'v1_100': 'ghostnet_100.in1k',
        'v1_130': 'ghostnet_130.in1k',
        'v2_100': 'ghostnetv2_100.in1k',
        'v2_160': 'ghostnetv2_160.in1k',
    }
