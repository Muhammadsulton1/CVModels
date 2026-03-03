from torch.nn import CrossEntropyLoss
from cnn_backbone.feature_extractor import FeatureExtractor
from cnn_backbone.trainer import Trainer
from torchvision import datasets, transforms
from utils.singeleton_config import ConfigReader

if __name__ == '__main__':
    cfg = ConfigReader()

    extractor = cfg.get('MODEL', 'extractor', 'deit_small_patch16')
    variant = cfg.get('MODEL', 'variant', 'small_reg')
    input_dim = int(cfg.get('MODEL', 'input_dim', 3))
    output_dim = int(cfg.get('MODEL', 'output_dim', 3))
    use_clf = bool(cfg.get('MODEL', 'use_clf', True))

    input_size = cfg.get('MODEL', 'input_size', [224, 224])
    if isinstance(input_size, (list, tuple)):
        input_size = tuple(int(x) for x in input_size)
    else:
        input_size = (int(input_size[0]), int(input_size[1]))

    dataset_root = cfg.get('DATASET', 'root', 'dataset')

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    criterion = CrossEntropyLoss(reduction='mean')

    fe = FeatureExtractor()
    model = fe.extract_features(
        model_name=str(extractor),
        input_dim=input_dim,
        output_dim=output_dim,
        clf_mode=use_clf
    )

    trainer = Trainer(model, criterion, dataset)
    trainer.fit()
