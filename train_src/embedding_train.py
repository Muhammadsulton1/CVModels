"""Пример обучения эмбеддингов через ``TrainerEmbeddings`` / ``EmbeddingTrainer``.

Требуются ``config/train_conf.yaml``, папки с классами, пакет ``emb_loss``.

Пайплайн изображений: все Albumentations задаются **здесь**, в ``transform=`` у
``datasets.ImageFolder`` (train отдельно от val). ``TrainerEmbeddings`` только батчит
тензоры; в него передаётся ``train_transform=None``, ``val_transform=None``, чтобы не
включать второй проход через ``TransformWrapper`` в ``build_dataloaders``.

Запуск из корня репозитория:

    py train_src/embedding_train.py

Train/val — отдельные каталоги ``<DATASET.root>/train`` и ``.../val`` (``ImageFolder``), без ``random_split`` в коде.

"""
import os
import sys

from torchvision import datasets
from emb_loss.losses import ArcCenterLoss, ArcLoss, CenterArcLoss, CenterLoss, SupConLoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_SRC = os.path.join(PROJECT_ROOT, 'models_src')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_SRC)

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from models_src.emb_model import EmbeddingModel
from models_src.get_feature import ModelFactory
from train_src.trainer import TrainerEmbeddings
from utils.singeleton_config import ConfigReader


class AlbumentationTransform:
    """Обёртка: PIL → numpy → Albumentations → tensor для ``ImageFolder(transform=...)``."""

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


def build_criterion(cfg: ConfigReader, embedding_dim: int, num_classes: int):
    loss_name = str(cfg.get('EMBEDDING_LOSS', 'name', 'arc_center')).lower()
    arc_s = float(cfg.get('EMBEDDING_LOSS', 'arc_s', 64.0))
    arc_m = float(cfg.get('EMBEDDING_LOSS', 'arc_m', 0.5))

    if loss_name == 'supcon':
        return SupConLoss(temperature=float(cfg.get('EMBEDDING_LOSS', 'temperature', 0.07)))
    if loss_name == 'arc':
        return ArcLoss(in_features=embedding_dim, num_classes=num_classes, s=arc_s, m=arc_m)
    if loss_name == 'center':
        return CenterLoss(num_classes=num_classes, feat_dim=embedding_dim)
    if loss_name == 'arc_center':
        return ArcCenterLoss(
            in_features=embedding_dim,
            num_classes=num_classes,
            center_weight=float(cfg.get('EMBEDDING_LOSS', 'center_weight', 0.05)),
            s=arc_s,
            m=arc_m,
        )
    if loss_name == 'center_arc':
        return CenterArcLoss(
            in_features=embedding_dim,
            num_classes=num_classes,
            arc_weight=float(cfg.get('EMBEDDING_LOSS', 'arc_weight', 0.05)),
            s=arc_s,
            m=arc_m,
        )
    raise ValueError(
        f'Unknown EMBEDDING_LOSS.name={loss_name}. Use: supcon, arc, center, arc_center, center_arc.'
    )


if __name__ == '__main__':
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
            f'Train/val class mappings differ: train={train_ds.class_to_idx}, val={val_ds.class_to_idx}'
        )

    config['DATASET']['num_classes'] = len(train_ds.classes)
    config['METRICS']['class_names'] = list(train_ds.classes)

    backbone, _ = ModelFactory(config).build()
    model = EmbeddingModel(config, backbone)
    criterion = build_criterion(cfg, model.embedding_dim, len(train_ds.classes))

    trainer = TrainerEmbeddings(model, criterion, train_ds, val_ds)

    trainer.fit()
