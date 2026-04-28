import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcLoss(nn.Module):
    """
    ArcLoss: Additive Angular Margin Loss (CVPR 2019).
    Содержит обучаемые веса классификатора — модель возвращает просто эмбеддинг.

    Args:
        in_features:  размерность эмбеддинга (embed_dim)
        num_classes:  кол-во известных классов (3: металл, руда, дерево)
        s:            scale — масштаб логитов, рекомендуется 32–64
        m:            угловой марджин в радианах, рекомендуется 0.3–0.5
    """

    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m

        # Предвычисляем тригонометрию один раз — не на каждый forward
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # Easy Margin: порог cos(π - m) и поправка sin(π - m) * m
        # Нужны чтобы phi не уходил вниз когда θ близко к π
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # Веса классификатора — "прототипы" классов на гиперсфере
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-нормализованные эмбеддинги (B, in_features)
            labels:     целочисленные метки классов (B,), значения 0..num_classes-1
        """
        # cos(θ) = (emb · W^T) при нормализованных векторах
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))

        # sin(θ) через тождество Пифагора, clamp защищает от sqrt отрицательного
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-7))

        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Easy Margin: если θ большой (cosine < threshold),
        # cos(θ + m) заменяем на линейную аппроксимацию
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # Применяем марджин только к целевому классу
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return F.cross_entropy(logits * self.s, labels)

    @torch.no_grad()
    def get_cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Только косинусы без марджина — для инференса и определения порога."""
        return F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))


class CenterLoss(nn.Module):
    """
    Center Loss (ECCV 2016).
    Минимизирует расстояние от эмбеддинга до центроида его класса.

    Args:
        num_classes: кол-во известных классов
        feat_dim:    размерность эмбеддинга (должна совпадать с ArcFaceLoss.in_features)
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-нормализованные эмбеддинги (B, feat_dim)
            labels:     метки классов (B,)
        """
        # Нормализуем центроиды — они должны жить на той же гиперсфере что и эмбеддинги
        centers_norm = F.normalize(self.centers, dim=1)

        # Эффективное евклидово расстояние через ||a-b||² = ||a||² + ||b||² - 2a·b
        # Поскольку эмбеддинги нормализованы, ||a||² = 1 и ||b||² = 1,
        # поэтому ||a-b||² = 2 - 2·cos(θ) — это и есть косинусное расстояние × 2
        batch_size = embeddings.size(0)
        distmat = (
            torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(centers_norm, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(embeddings, centers_norm.t(), beta=1, alpha=-2)

        # Маска: для каждого примера берём расстояние только до его класса
        classes = torch.arange(self.num_classes, device=embeddings.device)
        mask = labels.unsqueeze(1).expand(batch_size, self.num_classes).eq(
            classes.expand(batch_size, self.num_classes)
        )

        loss = distmat[mask].clamp(min=1e-12).mean()
        return loss


class ArcCenterLoss(nn.Module):
    """ArcFace as the main separator plus a small CenterLoss compactness term."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            center_weight: float = 0.05,
            s: float = 64.0,
            m: float = 0.5,
    ):
        super().__init__()
        self.arc = ArcLoss(in_features=in_features, num_classes=num_classes, s=s, m=m)
        self.center = CenterLoss(num_classes=num_classes, feat_dim=in_features)
        self.center_weight = center_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.arc(embeddings, labels) + self.center_weight * self.center(embeddings, labels)


class CenterArcLoss(nn.Module):
    """CenterLoss as the main term plus a small ArcFace separation term."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            arc_weight: float = 0.05,
            s: float = 64.0,
            m: float = 0.5,
    ):
        super().__init__()
        self.center = CenterLoss(num_classes=num_classes, feat_dim=in_features)
        self.arc = ArcLoss(in_features=in_features, num_classes=num_classes, s=s, m=m)
        self.arc_weight = arc_weight

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.center(embeddings, labels) + self.arc_weight * self.arc(embeddings, labels)


class SupConLoss(nn.Module):
    """Supervised contrastive loss for batches shaped as (image, class_id)."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        logits = embeddings @ embeddings.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
        logits_mask = ~eye

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor = positives_per_anchor > 0
        if not valid_anchor.any():
            return embeddings.sum() * 0.0

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1)
        return -mean_log_prob_pos[valid_anchor].mean()