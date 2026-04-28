import torch
import torch.nn as nn
import timm


def _freeze_module_params(module: nn.Module) -> None:
    """
        Замораживает все параметры переданного модуля.

        Устанавливает ``requires_grad = False`` для каждого тензора параметров,
        что полностью исключает их из графа вычислений и обновления оптимизатором.

        Используется совместно с ``module.eval()`` при заморозке backbone:
        - ``requires_grad=False`` — нет обновления весов
        - ``eval()``              — детерминированный forward (BN использует
                                    running_mean/var, Dropout отключён)

        Аргументы:
            module (nn.Module): любой модуль PyTorch — backbone, слой, подсеть.

        Пример:
            >>> _freeze_module_params(model.backbone)
            >>> # Проверка:
            >>> any(p.requires_grad for p in model.backbone.parameters())
            False
    """
    for p in module.parameters():
        p.requires_grad = False


class _TimmExtractorBase(nn.Module):
    """
    Базовый класс для всех backbone-экстракторов на основе библиотеки timm.

    Инкапсулирует единую логику:
      - загрузку pretrained-весов через ``timm.create_model``
      - адаптацию первого Conv2d к произвольному числу каналов (``in_chans``)
      - опциональную заморозку весов backbone
      - автоматическое определение реального размера выходных признаков
        через dummy-проход (решает проблему расхождения ``backbone.num_features``
        и фактического выхода у MobileNetV3/V4, EfficientNet и др.)
      - унифицированную агрегацию признаков для CNN и Transformer-архитектур

    Создание конкретного экстрактора
    ─────────────────────────────────
    Подклассу достаточно объявить словарь ``_MODELS``:

        class ResNetExtractor(_TimmExtractorBase):
            _MODELS = {
                'resnet50': 'resnet50.a1_in1k',
            }

    ``ModelFactory`` затем динамически комбинирует миксин с этим классом
    через ``type()``, так что ``_MODELS`` нужного backbone подставляется
    в MRO автоматически.

    Атрибуты класса:
        _MODELS (dict[str, str]): Словарь ``{alias: timm_model_name}``.
            Пустой в базовом классе — обязательно переопределяется в подклассах.

    Атрибуты экземпляра:
        backbone (nn.Module): timm-модель без классификационной головы
            (``num_classes=0``). forward() возвращает признаки.
        num_features (int): Реальный размер выходного вектора признаков,
            определённый через dummy-проход. Используется для построения
            голов классификации / проекторов.
        CONFIG: Оригинальный конфиг, переданный при инициализации.

    Конфигурация (CONFIG['MODEL']):
        size (str):
            Alias из словаря ``_MODELS`` подкласса.
            Пример: ``'resnet50'``, ``'small_reg'``, ``'base'``.

        input_dims (int):
            Число каналов входного изображения.
            timm автоматически адаптирует первый Conv2d:
            - ``input_dims < 3``: усредняет pretrained-веса по каналам
            - ``input_dims > 3``: повторяет/дополняет нулями

        img_size (list[int]):
            Размер изображения ``[H, W]``, используется при dummy-проходе
            для корректного определения ``num_features`` у архитектур
            с зависящим от разрешения размером выхода.

        pretrained (bool, optional):
            Загружать ли pretrained-веса. По умолчанию ``True``.

        freeze_backbone (bool, optional):
            Заморозить ли backbone (``requires_grad=False``).
            По умолчанию ``True``. При заморозке обязательно
            переопределите ``train()`` в вышестоящей модели, чтобы
            backbone оставался в ``eval()``-режиме во время обучения::

                def train(self, mode=True):
                    super().train(mode)
                    self.backbone.eval()   # BN всегда использует running stats
                    return self

        return_cls_token (bool, optional):
            Только для Transformer-архитектур (ViT, DINOv2, DeiT и др.).
            - ``True``  — возвращает CLS-токен ``features[:, 0]`` → (B, C)
            - ``False`` — возвращает mean-pool по всем токенам → (B, C)
            По умолчанию ``True``. Для DINOv2 рекомендуется ``True``.

    Примечание о num_features:
        ``backbone.num_features`` у ряда архитектур указывает на
        промежуточный размер (до финального expansion-слоя), тогда как
        реальный выход при ``num_classes=0`` может быть другим.
        Пример: MobileNetV3-small → ``backbone.num_features=576``,
        реальный выход → ``1024``. Поэтому ``num_features`` всегда
        определяется через реальный dummy-проход, а не через атрибут модели.
    """
    _MODELS: dict[str, str] = {}

    def __init__(self, CONFIG) -> None:
        super().__init__()
        self.CONFIG = CONFIG

        SIZE = CONFIG['MODEL']['size']
        INPUT_DIMS = CONFIG['MODEL']['input_dims']
        PRETRAINED = CONFIG['MODEL'].get('pretrained', True)
        FREEZE_BACKBONE = CONFIG['MODEL'].get('freeze_backbone', True)

        if SIZE not in self._MODELS:
            raise ValueError(
                f"[{type(self).__name__}] Неизвестный вариант '{SIZE}'. "
                f"Доступные: {list(self._MODELS)}"
            )

        self.backbone = timm.create_model(
            self._MODELS[SIZE],
            pretrained=PRETRAINED,
            num_classes=0,
            in_chans=INPUT_DIMS,
        )
        if FREEZE_BACKBONE:
            _freeze_module_params(self.backbone)

        # Определяем num_features через _extract, а не через backbone.num_features,
        # так как у CNN вроде MobileNetV3/V4 и EfficientNet атрибут num_features
        # указывает на размер ДО финального expansion Conv, тогда как реальный
        # выход при num_classes=0 может быть другим.
        self.num_features = self._infer_num_features(INPUT_DIMS)

    def _infer_num_features(self, input_dims: int) -> int:
        """
        Определяет реальный размер вектора признаков через dummy-проход.

        Переводит backbone в ``eval()`` на время проверки, чтобы
        BatchNorm использовал running-статистику и не требовал batch_size > 1.
        После возвращает backbone в исходное состояние.

        Аргументы:
            input_dims (int): Число каналов входного изображения.

        Возвращает:
            int: Размер последней оси выходного тензора ``(B, C)`` → C.

        Пример:
            DINOv2 small_reg (518×518):
                forward_features → (1, 1370, 384) → CLS → (1, 384) → C=384

            MobileNetV3-small (224×224):
                forward_features → (1, 1024) → C=1024
                (backbone.num_features=576 — неверно, реальный выход 1024)
        """
        img_h, img_w = self.CONFIG['MODEL']['img_size']
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, input_dims, img_h, img_w)
            out = self._extract(dummy)
        return out.shape[1]

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлекает вектор признаков фиксированного размера из входного тензора.

        Обрабатывает три топологии выходов timm при ``num_classes=0``:

        CNN (ResNet, MobileNet, EfficientNet, ConvNeXt, ...):
            ``forward_features(x)`` → ``(B, C, H, W)``
            → Global Average Pooling по пространственным осям
            → ``(B, C)``

        Transformer (ViT, DINOv2, DeiT, Swin, ...):
            ``forward_features(x)`` → ``(B, N+1, C)``
            где N — число патчей, +1 — CLS-токен (позиция 0)

            При ``return_cls_token=True``  → ``features[:, 0]``  → ``(B, C)``
            При ``return_cls_token=False`` → ``features.mean(1)`` → ``(B, C)``

            Для DINOv2 и моделей с register-токенами рекомендуется CLS
            (``return_cls_token=True``), так как он обучен нести
            глобальное представление изображения.

        Уже пулированный вектор (dim == 2):
            ``(B, C)`` → без изменений

        Аргументы:
            x (torch.Tensor): Батч изображений ``(B, C_in, H, W)``.

        Возвращает:
            torch.Tensor: Матрица признаков ``(B, num_features)``.
        """
        features = self.backbone.forward_features(x)

        if features.dim() == 4:
            # CNN: (B, C, H, W) → Global Average Pooling
            features = features.mean([-2, -1])

        elif features.dim() == 3:
            if self.CONFIG['MODEL'].get('return_cls_token', True):
                features = features[:, 0]
            else:
                features = features.mean(dim=1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Точка входа для ``model(x)``.

        Делегирует вызов в ``_extract``, что позволяет подклассам
        (EmbeddingExtractor, ClassificationExtractor) переопределить
        только ``forward``, не трогая логику агрегации признаков.

        Аргументы:
            x (torch.Tensor): Батч изображений ``(B, C_in, H, W)``.

        Возвращает:
            torch.Tensor: ``(B, num_features)`` — сырые признаки backbone.
        """
        return self._extract(x)
