import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.nn import CrossEntropyLoss
from backbone import FeatureExtractor
from trainer import Trainer
from torchvision import datasets
from utils.singeleton_config import ConfigReader


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


if __name__ == '__main__':
    cfg = ConfigReader()

    extractor = cfg.get('MODEL', 'extractor', 'deit_small_patch16')
    variant = cfg.get('MODEL', 'variant', 'small_reg')
    input_dim = int(cfg.get('MODEL', 'input_dim', 3))
    output_dim = int(cfg.get('MODEL', 'output_dim', 3))
    use_clf = bool(cfg.get('MODEL', 'use_clf', True))

    input_size = cfg.get('MODEL', 'input_size', [224, 224])
    if isinstance(input_size, (list, tuple)):
        h, w = int(input_size[0]), int(input_size[1])
    else:
        h, w = int(input_size[0]), int(input_size[1])

    dataset_root = cfg.get('DATASET', 'root', 'dataset')

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

    dataset = datasets.ImageFolder(root=dataset_root, transform=None)
    criterion = CrossEntropyLoss(reduction='mean')

    fe = FeatureExtractor()
    model = fe.get_model(
        model_name=str(extractor),
        size=str(variant),
        input_dim=input_dim,
        output_dim=output_dim,
        clf_mode=use_clf
    )

    trainer = Trainer(model, criterion, dataset,
                     train_transform=train_transform, val_transform=val_transform)
    trainer.fit()
