"""
Референс для metric learning: `backbone` из `ModelFactory` + проектор + L2 по строкам.

**Шаблон для своего класса**: скопировать/расширить; в `train_src/embedding_train.py`
подставить ваш `nn.Module`. Для типичных лоссов (SupCon, Arc…) ожидается выход
`(N, D)` и нормализация `|| · ||₂ = 1` на батче (если лосс считает косинусы).

Обязательно задайте числовой атрибут `embedding_dim` (размер D), если тренер/лосс его читают из модели.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from get_feature import ModelFactory


class EmbeddingModel(nn.Module):
    """
    Backbone (замороженный) + projection head + L2-нормализация.

    Output: (B, embedding_dim), ||output||₂ = 1

    CONFIG['MODEL']:
        hidden_dim     — размер embedding (опц., по умолчанию num_features // 2)
        head_dropout   — dropout в проекторе (опц., по умолчанию 0.1)
    """

    def __init__(self, config: DictConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        cfg = config['MODEL']
        hidden_dim = cfg.get('hidden_dim', None)
        EMBEDDING_DIM = max(1, hidden_dim if hidden_dim is not None else self.num_features // 2)
        self.embedding_dim = int(EMBEDDING_DIM)
        DROPOUT = cfg.get('head_dropout', 0.1)

        self.projector = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LayerNorm(self.num_features // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(self.num_features // 2, EMBEDDING_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projected = self.projector(features)
        return F.normalize(projected, p=2, dim=1)


if __name__ == '__main__':
    config = OmegaConf.load('../config/train_conf.yaml')
    dummy = torch.randn(1, 3, 518, 518)

    backbone, transform = ModelFactory(config).build()
    model = EmbeddingModel(config, backbone)
    model.eval()

    with torch.no_grad():
        out = model(dummy)

    print(f"output shape: {out.shape}")
    print(f"L2 norm: {out.norm(dim=1)}")
