"""Пример обучения классификации через ``TrainerClassification`` / ``ClassificationTrainer``.

Требуется ``config/train_conf.yaml`` и папка с классами (ImageFolder), путь в ``DATASET.root``.

Запуск из корня репозитория (Windows / Linux):

    cd /path/to/CVModels
    py train_src/classification_train.py

При необходимости отредактируйте ``config/train_conf.yaml`` (путь к данным, эпохи, lr).
"""
import os
import sys

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torchvision import datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_SRC = os.path.join(PROJECT_ROOT, 'models_src')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_SRC)

from omegaconf import OmegaConf

from models_src.clf_model import ModelClassification
from models_src.get_feature import ModelFactory
from train_src.trainer import TrainerClassification
from utils.logger import setup_logger
from utils.singeleton_config import ConfigReader

logger = setup_logger()


class AlbumentationTransform:
    """Обёртка: принимает PIL, возвращает tensor для совместимости с ImageFolder."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if hasattr(img, 'convert'):
            img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        return self.transform(image=img)['image']


def build_transforms(cfg: ConfigReader):
    input_size = cfg.get('MODEL', 'img_size', cfg.get('MODEL', 'input_size', [224, 224]))
    h, w = int(input_size[0]), int(input_size[1])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = AlbumentationTransform(A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]))

    val_transform = AlbumentationTransform(A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]))

    return train_transform, val_transform

def resolve_data_root(cfg: ConfigReader) -> str:
    root = cfg.get('DATASET', 'root', 'metric_learning_data')
    if not os.path.isabs(root):
        root = os.path.join(PROJECT_ROOT, root)
    return os.path.abspath(root)


def main():
    cfg = ConfigReader()
    config = cfg.raw_config

    train_transform, val_transform = build_transforms(cfg)
    data_root = resolve_data_root(cfg)

    train_ds = datasets.ImageFolder(
        root=os.path.join(data_root, 'train'),
        transform=train_transform,
    )
    val_ds = datasets.ImageFolder(
        root=os.path.join(data_root, 'val'),
        transform=val_transform,
    )

    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(
            'train и val ImageFolder: разные class_to_idx. Проверьте порядок папок классов в train/ и val/.'
        )

    n_cls = len(train_ds.classes)
    config['DATASET']['num_classes'] = n_cls
    if 'MODEL' not in config or config['MODEL'] is None:
        config['MODEL'] = {}
    config['MODEL']['output_dims'] = n_cls

    names_cfg = cfg.get('METRICS', 'class_names', None)
    if names_cfg:
        names_list = OmegaConf.to_container(names_cfg, resolve=True)
        if isinstance(names_list, str):
            names_list = [names_list]
        if list(names_list) != list(train_ds.classes):
            logger.warning(
                'METRICS.class_names не совпадает с train_ds.classes — подставляем имена из train.'
            )
    if 'METRICS' not in config or config['METRICS'] is None:
        config['METRICS'] = {}
    config['METRICS']['class_names'] = list(train_ds.classes)

    backbone, _ = ModelFactory(cfg.raw_config).build()
    model = ModelClassification(cfg.raw_config, backbone)
    criterion = nn.CrossEntropyLoss()

    trainer = TrainerClassification(
        model, criterion, train_ds, val_ds)
    trainer.fit()


if __name__ == '__main__':
    main()
