from typing import Any
from omegaconf import DictConfig

import timm
import torch
from torch import nn

from models_src.encoder import _TimmExtractorBase
from models_src.models import ResNetExtractor, MobileNetV2Extractor, MobileNetV3Extractor, MobileNetV4Extractor, \
    ShuffleNetV2Extractor, EfficientNetExtractor, EfficientNetV2Extractor, RegNetExtractor, ConvNextExtractor, \
    ConvNeXtV2Extractor, SwinTransformerExtractor, SwinV2Extractor, MaxViTExtractor, ViTExtractor, DeiTExtractor, \
    DeiT3Extractor, DINOv2Extractor, EfficientViTExtractor, MobileViTExtractor, MobileViTV2Extractor, RepViTExtractor, \
    FastViTExtractor, GhostNetExtractor


class ModelFactory:
    """
    Фабрика backbone-экстракторов на основе timm.

    Динамически собирает нужный класс из двух компонентов:
      - backbone-миксин (содержит словарь ``_MODELS`` с timm-тегами)
      - ``_TimmExtractorBase`` (логика загрузки, заморозки, агрегации признаков)

    Сборка выполняется через ``type()`` с MRO:
        CombinedCls → backbone_mixin → _TimmExtractorBase → nn.Module

    Миксин идёт первым, поэтому его ``_MODELS`` перекрывает пустой ``{}``
    базового класса. Это позволяет добавлять новые backbone-ы, не трогая
    логику обучения — достаточно зарегистрировать новый миксин в ``_backbone_registry``.

    Использование:
        model, transform = ModelFactory(CONFIG).build()

        # Полный пример:
        CONFIG = OmegaConf.load('config/train_conf.yaml')
        model, transform = ModelFactory(CONFIG).build()

        img = transform(pil_image).unsqueeze(0)   # (1, C, H, W)
        features = model(img)                     # (1, num_features)

    Конфигурация (CONFIG['MODEL']):
        backbone (str):
            Ключ из ``_backbone_registry``.
            Регистронезависим — преобразуется в lower() при инициализации.
            Пример: ``'dinov2'``, ``'resnet'``, ``'mobilenetv4'``.

        size (str):
            Alias из словаря ``_MODELS`` выбранного backbone.
            Пример для DINOv2: ``'small_reg'``, ``'base'``, ``'large'``.
            Пример для ResNet: ``'resnet50'``, ``'resnet18'``.

        input_dims (int):
            Число каналов входного изображения.
            timm адаптирует первый Conv2d автоматически:
            - 1 канал (grayscale): усредняет pretrained RGB-веса
            - 4+ каналов: повторяет / дополняет нулями

        img_size (list[int]):
            Размер изображения ``[H, W]``.
            Используется при dummy-проходе для определения ``num_features``.
            Должен совпадать с реальным размером батча при обучении.

        pretrained (bool, optional):
            Загружать ли pretrained-веса с HuggingFace Hub / timm.
            По умолчанию ``True``.
            При ``False`` — случайная инициализация (для отладки или обучения с нуля).

        freeze_backbone (bool, optional):
            Заморозить ли веса backbone (``requires_grad=False``).
            По умолчанию ``True``.

            .. warning::
                При заморозке необходимо явно переопределить ``train()``
                в модели-обёртке (EmbeddingModel, ModelClassification),
                чтобы backbone оставался в ``eval()``-режиме во время обучения.
                Иначе ``BatchNorm`` будет считать статистику по батчу,
                а не использовать ``running_mean/var``::

                    def train(self, mode=True):
                        super().train(mode)
                        self.backbone.eval()
                        return self

        return_cls_token (bool, optional):
            Только для Transformer-архитектур (ViT, DINOv2, DeiT, Swin, ...).
            - ``True``  → CLS-токен ``features[:, 0]`` → ``(B, C)``
            - ``False`` → mean-pool по токенам → ``(B, C)``
            По умолчанию ``True``.
            Для DINOv2 настоятельно рекомендуется ``True`` —
            CLS-токен обучен нести глобальное представление изображения.

    Пример конфига (train_conf.yaml):
        MODEL:
          backbone: dinov2
          size: small_reg
          pretrained: true
          freeze_backbone: true
          return_cls_token: true
          input_dims: 3
          img_size: [518, 518]

    Возвращает (метод build):
        model (nn.Module):
            Готовый экстрактор в ``eval()``-режиме.
            ``model(x)`` → ``(B, num_features)``.
            ``model.num_features`` — реальный размер вектора признаков.

        transform (Callable):
            torchvision-совместимый препроцессинг, построенный по
            ``timm.data.resolve_model_data_config`` — нормализация,
            resize, crop строго соответствуют тому, как модель обучалась.

    Атрибуты класса:
        _backbone_registry (dict[str, type]):
            Реестр всех доступных backbone-миксинов.
            Ключ — строковый alias (используется в конфиге).
            Значение — класс-миксин с заполненным ``_MODELS``.

            Для добавления нового backbone достаточно:
            1. Создать миксин с ``_MODELS`` в ``models_src/models.py``
            2. Зарегистрировать его здесь::

                class MyBackboneExtractor:
                    _MODELS = {'variant': 'timm_model_tag'}

                ModelFactory._backbone_registry['mybackbone'] = MyBackboneExtractor
    """

    _backbone_registry: dict[str, type] = {
        'resnet': ResNetExtractor,
        'mobilenetv2': MobileNetV2Extractor,
        'mobilenetv3': MobileNetV3Extractor,
        'mobilenetv4': MobileNetV4Extractor,
        'shufflenetv2': ShuffleNetV2Extractor,
        'efficientnet': EfficientNetExtractor,
        'efficientnetv2': EfficientNetV2Extractor,
        'regnet': RegNetExtractor,
        'convnext': ConvNextExtractor,
        'convnextv2': ConvNeXtV2Extractor,
        'swin': SwinTransformerExtractor,
        'swinv2': SwinV2Extractor,
        'maxvit': MaxViTExtractor,
        'vit': ViTExtractor,
        'deit': DeiTExtractor,
        'deit3': DeiT3Extractor,
        'dinov2': DINOv2Extractor,
        'efficientvit': EfficientViTExtractor,
        'mobilevit': MobileViTExtractor,
        'mobilevitv2': MobileViTV2Extractor,
        'repvit': RepViTExtractor,
        'fastvit': FastViTExtractor,
        'ghostnet': GhostNetExtractor,
    }

    def __init__(self, CONFIG: DictConfig) -> None:
        self.config = CONFIG
        self.model_name = CONFIG['MODEL']['backbone'].lower()

    def build(self) -> tuple[nn.Module, Any]:
        """
        Собирает и возвращает backbone-экстрактор с препроцессингом.

        Алгоритм:
            1. Валидирует ``backbone`` из конфига по ``_backbone_registry``.
            2. Динамически создаёт класс через ``type()``:
               ``(backbone_mixin, _TimmExtractorBase)`` — миксин первым,
               чтобы его ``_MODELS`` перекрыл пустой словарь базового класса.
            3. Инициализирует модель, переводит в ``eval()``.
            4. Строит препроцессинг через ``timm.data.resolve_model_data_config``
               на основе конфига самой timm-модели (нормализация, размер и т.д.)

        Возвращает:
            tuple[nn.Module, Callable]:
                - model: готовый экстрактор, ``model(x)`` → ``(B, num_features)``
                - transform: препроцессинг, совместимый с torchvision

        Исключения:
            KeyError: если ``backbone`` не найден в ``_backbone_registry``.
                Сообщение содержит список всех доступных backbone-ов.
        """
        if self.model_name not in self._backbone_registry:
            raise KeyError(
                f"Backbone '{self.model_name}' не найден.\n"
                f"Доступные: {list(self._backbone_registry)}"
            )

        backbone_mixin = self._backbone_registry[self.model_name]
        combined_cls = type(
            backbone_mixin.__name__,
            (backbone_mixin, _TimmExtractorBase),
            {},
        )

        model = combined_cls(self.config)
        model.eval()

        # Препроцессинг строится из конфига самой timm-модели:
        # нормализация (mean/std), resize, crop_pct строго соответствуют обучению
        data_config = timm.data.resolve_model_data_config(model.backbone)
        transform = timm.data.create_transform(**data_config, is_training=False)

        return model, transform


if __name__ == '__main__':
    from omegaconf import OmegaConf, DictConfig

    CONFIG = OmegaConf.load('../config/train_conf.yaml')
    dummy = torch.randn(1, 3, 518, 518)

    model, transform = ModelFactory(CONFIG).build()
    out = model(dummy)
    print(f"backbone : {CONFIG['MODEL']['backbone']} / {CONFIG['MODEL']['size']}")
    print(f"output   : {out.shape}")
    print(f"features : {out}")