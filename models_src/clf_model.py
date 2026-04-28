import torch
from omegaconf import OmegaConf
from torch import nn

from get_feature import ModelFactory


class ModelClassification(nn.Module):
    def __init__(self, config, backbone) -> None:
        super().__init__()

        self.CONFIG = config
        self.backbone = backbone
        self.num_features = backbone.num_features

        m_cfg = self.CONFIG.get('MODEL', {}) or {}

        OUTPUT_DIMS = m_cfg.get('output_dims')
        if OUTPUT_DIMS is None:
            raise ValueError(
                "CONFIG['MODEL']['output_dims'] обязателен для ClassificationModel.\n"
                f"Укажите число классов: output_dims: <N>"
            )
        if int(OUTPUT_DIMS) < 1:
            raise ValueError(
                "CONFIG['MODEL']['output_dims'] должно быть целым числом классов ≥ 1.\n"
                f"Текущее значение: {OUTPUT_DIMS!r}"
            )

        hidden_dim = m_cfg.get('hidden_dim', None)
        HIDDEN_DIM = max(1, hidden_dim if hidden_dim is not None else self.num_features // 2)
        dropout_p = float(m_cfg.get('head_dropout', 0.3))

        self.head = nn.Sequential(
            nn.Linear(self.num_features, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIMS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


if __name__ == '__main__':
    CONFIG = OmegaConf.load('../config/train_conf.yaml')
    if int(CONFIG.MODEL.get('output_dims', 0) or 0) < 1:
        CONFIG.MODEL.output_dims = 3
    dummy = torch.randn(1, 3, 518, 518)

    model, transforms = ModelFactory(CONFIG).build()

    clf = ModelClassification(CONFIG, model)
    clf.eval()

    with torch.no_grad():
        out = clf(dummy)
    print(f"output shape : {out.shape}")
