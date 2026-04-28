import os
import numpy as np
import torch

from torch.utils.data import Dataset
from utils.logger import setup_logger


logger = setup_logger()


class TransformWrapper(Dataset):
    """Обёртка над датасетом: применяет ``transform(x)`` к элементу перед выдачей в DataLoader.

    Подключается из ``BaseTrainer.build_dataloaders``, только если переданы непустые
    ``train_transform`` / ``val_transform``. Два допустимых варианта (не смешивайте):

    * Трансформы задаёте прямо в ``ImageFolder(...)`` → в тренер передайте оба значения как
      ``None`` (см. ``train_src/embedding_train.py``).
    * ``ImageFolder(..., transform=None)`` подаёт только PIL → тогда задаёте
      ``train_transform`` / ``val_transform`` здесь. Двойная подача одного и того же
      даёт второй проход по тензору и ошибки в Albumentations.
    """

    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
            if self.transform is not None:
                x = self.transform(x)
            return x, y
        if self.transform is not None:
            return self.transform(item)
        return item


def _embedding_retrieval_metrics_at_k(
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Recall@k / Precision@k over all samples (nearest-neighbor retrieval, exclude self).

    Recall@k: share of queries for which ≥1 neighbor of the same class appears in top‑k
    positions by cosine similarity (embeddings assumed L2-normalized or normalized here).

    Precision@k: mean over queries of (same‑class neighbors in top‑k) / k.

    Queries whose class occurs only once in the split are skipped so metrics stay defined.

    Implemented with row-wise chunks to avoid materializing full N×N similarity on RAM.
    """
    out: dict[str, float] = {}
    n = embeddings.size(0)
    if n < 3:
        return out

    device = embeddings.device
    z = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    lab_all = labels.view(-1).long().to(device)

    n_cls = int(lab_all.max().item()) + 1
    counts = torch.bincount(lab_all.cpu(), minlength=n_cls)

    recall_hits = {int(k): 0 for k in ks}
    prec_acc = {int(k): 0.0 for k in ks}
    valid_q = 0

    kk_max = int(max(ks))
    kk_max_eff = min(kk_max, n - 1)
    if kk_max_eff < 1:
        return out

    chunk_rows = min(512, max(1, n))

    for qs in range(0, n, chunk_rows):
        qe = min(n, qs + chunk_rows)
        q_rows = qe - qs
        block = z[qs:qe] @ z.t()
        rr = torch.arange(qs, qe, device=device)
        block[torch.arange(q_rows, device=device), rr] = -float('inf')

        for qi in range(q_rows):
            i_global = qs + qi
            yi = int(lab_all[i_global].item())
            if counts[yi].item() < 2:
                continue

            row = block[qi]
            ordered = torch.argsort(row, descending=True)
            neighbors = ordered[:kk_max_eff]

            valid_q += 1
            for kk in ks:
                ki = min(int(kk), kk_max_eff)
                if ki < 1:
                    continue
                nb_idx = neighbors[:ki]
                same = (lab_all[nb_idx] == yi).sum().item()
                if same > 0:
                    recall_hits[int(kk)] += 1
                prec_acc[int(kk)] += same / float(ki)

    if valid_q == 0:
        return out

    vq = float(valid_q)
    for kk in ks:
        ki = int(kk)
        out[f'recall_at_{ki}'] = recall_hits[ki] / vq
        out[f'precision_at_{ki}'] = prec_acc[ki] / vq
    return out


def _plot_class_centroids_2d(
        centroid_matrix: np.ndarray,
        names: list[str],
        save_path: str,
        title: str = 'Embedding class centroids (2D PCA)',
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning('matplotlib not installed — skip centroid plot')
        return

    nc, _ = centroid_matrix.shape
    if nc < 2:
        logger.info('Skipping centroid plot — fewer than two class centroids')
        return

    x = centroid_matrix.astype(np.float64)
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(x)
    except ImportError:
        xc = x - x.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
        coords = xc @ vt.T[:, :2]

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    for c in range(nc):
        lbl = names[c] if c < len(names) else str(c)
        plt.scatter(
            coords[c, 0], coords[c, 1], s=130, alpha=0.9,
            color=cmap((c % 10) / 10.0), edgecolors='k', linewidths=0.5, label=str(lbl),
        )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    logger.info(f'Centroid plot saved → {save_path}')


def _plot_embedding_clusters_2d(
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroid_matrix: np.ndarray,
        names: list[str],
        save_path: str,
        title: str = 'Embedding clusters with class centroids (PCA2D)',
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning('matplotlib not installed — skip embedding cluster plot')
        return

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        logger.info('Skipping embedding cluster plot — need at least two embeddings')
        return
    if centroid_matrix.ndim != 2 or centroid_matrix.shape[0] < 1:
        logger.info('Skipping embedding cluster plot — no class centroids')
        return

    z = embeddings.astype(np.float64)
    c = centroid_matrix.astype(np.float64)
    labels = labels.reshape(-1).astype(np.int64)

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(z)
        centroid_coords = pca.transform(c)
    except ImportError:
        mean = z.mean(axis=0, keepdims=True)
        zc = z - mean
        _u, _s, vt = np.linalg.svd(zc, full_matrices=False)
        basis = vt[:2].T
        coords = zc @ basis
        centroid_coords = (c - mean) @ basis

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    plt.figure(figsize=(11, 8))
    cmap = plt.get_cmap('tab10')

    unique_labels = sorted(int(x) for x in np.unique(labels))
    for label in unique_labels:
        mask = labels == label
        name = names[label] if label < len(names) else str(label)
        color = cmap((label % 10) / 10.0)
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=18, alpha=0.45, color=color, label=f'{name} samples',
            edgecolors='none',
        )

    for idx, row in enumerate(centroid_coords):
        name = names[idx] if idx < len(names) else str(idx)
        color = cmap((idx % 10) / 10.0)
        plt.scatter(
            row[0], row[1],
            s=220, marker='X', alpha=0.98, color=color,
            edgecolors='black', linewidths=1.1, label=f'{name} centroid',
        )

    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close()
    logger.info(f'Embedding cluster plot saved → {save_path}')
