import os
import sys
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_MODELS_SRC = os.path.join(_PROJECT_ROOT, 'models_src')
for _p in (_PROJECT_ROOT, _MODELS_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models_src.clf_model import ModelClassification
from models_src.emb_model import EmbeddingModel
from models_src.get_feature import ModelFactory

from utils.drawer import PredictionVideoWriter
from utils.singeleton_config import ConfigReader

cfg = ConfigReader()


def _load_state_dict(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint,
                                                              dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def _input_hw(config: ConfigReader) -> tuple[int, int]:
    input_size = config.get('MODEL', 'img_size', config.get('MODEL', 'input_size', [224, 224]))
    return int(input_size[0]), int(input_size[1])


def _frame_to_tensor(
        frame,
        device: torch.device,
        input_w: int,
        input_h: int,
        mean: torch.Tensor,
        std: torch.Tensor,
) -> torch.Tensor:
    if frame is None:
        raise ValueError('Кадр пустой')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (input_w, input_h))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    tensor = tensor.div_(255.0)
    tensor = tensor.sub_(mean).div_(std)
    return tensor


class Predictor:
    """Классификация: чекпоинт от ``ModelClassification`` + тот же препроцесс, что при обучении."""

    def __init__(
            self,
            weights_path: str,
            window_size: int = 10,
            class_names: list[str] | None = None,
            smooth_mode: str = 'moda',
            config: ConfigReader | None = None,
    ) -> None:
        self._cfg = config or ConfigReader()
        OmegaConf.resolve(self._cfg.raw_config)

        self.class_names = class_names or []
        if not self.class_names:
            raise ValueError('Укажите class_names (число классов = len(class_names)).')

        self._cfg.raw_config['MODEL']['output_dims'] = len(self.class_names)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.prediction_window = deque(maxlen=window_size)

        self.smooth_mode = smooth_mode
        if self.smooth_mode not in ('moda', 'mean'):
            raise ValueError(f"Неизвестный smooth_mode: {self.smooth_mode}. Используй 'moda' или 'mean'")

        backbone, _ = ModelFactory(self._cfg.raw_config).build()
        self.model = ModelClassification(self._cfg.raw_config, backbone)

        _load_state_dict(self.model, weights_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        self.input_h, self.input_w = _input_hw(self._cfg)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def prepare_frame(self, frame):
        return _frame_to_tensor(frame, self.device, self.input_w, self.input_h, self.mean, self.std)

    @torch.no_grad()
    def predict(self, frame):
        x = self.prepare_frame(frame)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        return probs, pred_idx

    @staticmethod
    def _mode_last_tie(values) -> int:
        counts = Counter(values)
        max_count = max(counts.values())
        for v in reversed(values):
            if counts[v] == max_count:
                return v
        return values[-1]

    def smooth_mean(self, probs: torch.Tensor):
        self.prediction_window.append(probs.detach().cpu())
        stacked = torch.stack(list(self.prediction_window), dim=0)
        smooth_probs = stacked.mean(dim=0)
        smooth_idx = int(torch.argmax(smooth_probs).item())
        return smooth_probs, smooth_idx

    def smooth_moda(self, probs: torch.Tensor, pred_idx: int):
        self.prediction_window.append(pred_idx)
        window_list = list(self.prediction_window)
        smooth_idx = self._mode_last_tie(window_list)
        return probs.detach().cpu(), smooth_idx

    @torch.no_grad()
    def smooth_prediction(self, frame):
        t0 = time.perf_counter()
        probs, pred_idx = self.predict(frame)
        inference_time_s = time.perf_counter() - t0

        if self.smooth_mode == 'moda':
            smooth_probs, smooth_idx = self.smooth_moda(probs, pred_idx)
        else:
            smooth_probs, smooth_idx = self.smooth_mean(probs)

        inference_time_ms = inference_time_s * 1000
        inference_fps = 1000.0 / inference_time_ms if inference_time_ms > 0 else 0

        return {
            'probs': probs,
            'pred_idx': pred_idx,
            'pred_class': self.class_names[pred_idx],
            'smooth_probs': smooth_probs,
            'smooth_idx': smooth_idx,
            'smooth_class': self.class_names[smooth_idx],
            'inference_time_ms': inference_time_ms,
            'inference_fps': inference_fps,
        }

    def probs_to_dict(self, probs: torch.Tensor) -> dict[str, float]:
        probs = probs.detach().cpu().tolist()
        return {cls_name: float(prob) for cls_name, prob in zip(self.class_names, probs)}


class EmbeddingPredictor:
    """Эмбеддинги: чекпоинт от ``EmbeddingModel`` (как в ``embedding_train.py``).

    Вектор на выходе L2-нормирован (как в модели). Опционально — класс по ближайшему центроиду
    (строки ``class_centroids`` — усреднённые эмбеддинги классов, каждая строка с ||·||₂ = 1).
    """

    def __init__(
            self,
            weights_path: str,
            *,
            class_centroids: np.ndarray | None = None,
            class_names: list[str] | None = None,
            smooth_window: int = 1,
            config: ConfigReader | None = None,
    ) -> None:
        self._cfg = config or ConfigReader()
        OmegaConf.resolve(self._cfg.raw_config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        backbone, _ = ModelFactory(self._cfg.raw_config).build()
        self.model = EmbeddingModel(self._cfg.raw_config, backbone)
        _load_state_dict(self.model, weights_path, self.device)
        self.model.to(self.device)
        self.model.eval()

        self.input_h, self.input_w = _input_hw(self._cfg)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.class_names: list[str] = list(class_names) if class_names else []
        self._centroids: torch.Tensor | None = None
        if class_centroids is not None:
            c = np.asarray(class_centroids, dtype=np.float32)
            if c.ndim != 2:
                raise ValueError('class_centroids: ожидается форма (n_classes, embedding_dim)')
            c = torch.from_numpy(c).to(self.device)
            self._centroids = F.normalize(c, p=2, dim=1)
            if self.class_names and len(self.class_names) != self._centroids.size(0):
                raise ValueError(
                    f'len(class_names)={len(self.class_names)} != class_centroids.shape[0]={self._centroids.size(0)}'
                )

        self.smooth_window = max(1, int(smooth_window))
        self._emb_window: deque[torch.Tensor] = deque(maxlen=self.smooth_window)

    def prepare_frame(self, frame):
        return _frame_to_tensor(frame, self.device, self.input_w, self.input_h, self.mean, self.std)

    def prepare_tensor(
            self,
            tensor_chw: torch.Tensor,
            *,
            divide255_if_needed: bool = True,
            normalize: bool | None = None,
    ) -> torch.Tensor:
        """Батч (N,3,H,W): RGB tensor.

        По умолчанию вход распознаётся эвристикой:

        - uint8 / шкала 0–255 → ``/255``, затем ImageNet normalize как в обучении;
        - уже похожий на результат ``(x-mean)/std`` (float) → без повторной нормализации.

        Явно: ``normalize=True`` — всегда ImageNet normalize; ``normalize=False`` — не нормализовать
        (при ``divide255_if_needed`` деление на 255 сохраняется).
        """
        if tensor_chw.dim() == 3:
            x = tensor_chw.unsqueeze(0)
        else:
            x = tensor_chw

        x = x.to(self.device, dtype=torch.float32)

        forced = normalize is not None
        do_norm = True if normalize is None else bool(normalize)

        if divide255_if_needed and not forced:
            hi = float(x.detach().amax().cpu())
            lo = float(x.detach().amin().cpu())
            if x.dtype == torch.uint8 or hi > 1.75 or lo < -0.25:
                x = x.div_(255.0)
        elif divide255_if_needed and do_norm:
            hi = float(x.detach().amax().cpu())
            if x.dtype == torch.uint8 or hi > 1.75:
                x = x.div_(255.0)

        if normalize is None:
            ch_mean = x.detach().mean(dim=(0, 2, 3))
            mn = float(ch_mean.min().cpu())
            mx = float(ch_mean.max().cpu())
            hi2 = float(x.detach().amax().cpu())
            lo2 = float(x.detach().amin().cpu())
            if mn < -0.05 or mx > 0.15 or hi2 > 8.5 or lo2 < -8.5:
                do_norm = False

        if do_norm:
            x = x.sub_(self.mean).div_(self.std)

        return x

    @torch.no_grad()
    def embed(self, frame) -> torch.Tensor:
        """Один кадр BGR (OpenCV) → вектор эмбеддинга (D,) на CPU, float32."""
        x = self.prepare_frame(frame)
        z = self.model(x)[0].detach().float().cpu()
        return z

    @torch.no_grad()
    def embed_image_path(self, path: str) -> torch.Tensor:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f'Не удалось прочитать изображение: {path}')
        return self.embed(bgr)

    def smooth_embed(self, frame) -> torch.Tensor:
        """Скользящее среднее эмбеддингов по окну, затем L2-нормализация."""
        z = self.embed(frame)
        self._emb_window.append(z)
        stacked = torch.stack(list(self._emb_window), dim=0)
        z_mean = F.normalize(stacked.mean(dim=0), p=2, dim=0)
        return z_mean

    def reset_smooth(self) -> None:
        self._emb_window.clear()

    @torch.no_grad()
    def predict_with_centroids(self, frame, *, use_smooth: bool = False) -> dict:
        """Класс с максимальной косинусной близостью к центроидам (нужны ``class_centroids`` при init)."""
        if self._centroids is None:
            raise ValueError('Задайте class_centroids в конструкторе для predict_with_centroids')

        z = self.smooth_embed(frame) if use_smooth else self.embed(frame)
        z_d = z.to(self.device).unsqueeze(0)
        sims = (z_d @ self._centroids.t())[0]
        pred_idx = int(torch.argmax(sims).item())

        names = self.class_names
        out = {
            'embedding': z,
            'similarities': sims.detach().cpu(),
            'pred_idx': pred_idx,
            'pred_class': names[pred_idx] if names and pred_idx < len(names) else str(pred_idx),
        }
        if names:
            out['similarities_dict'] = {names[i]: float(sims[i].item()) for i in range(len(names))}
        return out

    @torch.no_grad()
    def predict_top_k(self, frame, *, k: int = 5, use_smooth: bool = False) -> dict:
        """Топ-K классов по косинусной близости к центроидам."""
        if self._centroids is None:
            raise ValueError('Задайте class_centroids в конструкторе')

        z = self.smooth_embed(frame) if use_smooth else self.embed(frame)
        z_d = z.to(self.device).unsqueeze(0)
        sims = (z_d @ self._centroids.t())[0]  # (n_classes,)

        k = min(k, sims.size(0))
        top_sims, top_idxs = torch.topk(sims, k=k)  # оба (k,)

        names = self.class_names
        top_k = [
            {
                'rank': i + 1,
                'pred_idx': int(idx.item()),
                'pred_class': names[int(idx.item())] if names else str(int(idx.item())),
                'similarity': float(sim.item()),
            }
            for i, (idx, sim) in enumerate(zip(top_idxs, top_sims))
        ]

        return {
            'embedding': z,
            'similarities': sims.detach().cpu(),
            'top_k': top_k,
            # первое место — для совместимости с predict_with_centroids
            'pred_idx': top_k[0]['pred_idx'],
            'pred_class': top_k[0]['pred_class'],
        }

    @torch.no_grad()
    def forward_timings(self, frame) -> dict:
        """embed + время (мс/FPS)."""
        t0 = time.perf_counter()
        z = self.embed(frame)
        dt = time.perf_counter() - t0
        ms = dt * 1000.0
        return {
            'embedding': z,
            'inference_time_ms': ms,
            'inference_fps': 1000.0 / ms if ms > 0 else 0.0,
        }


if __name__ == '__main__':
    # # Пример: классификация — чекпоинт от ModelClassification (пути поправьте под себя).
    # predictor = Predictor(
    #     weights_path=os.path.join(_PROJECT_ROOT, 'weights', 'DINOv2Extractor', 'best_checkpoint.pth'),
    #     window_size=30,
    #     class_names=['bad', 'good', 'normal'],
    #     smooth_mode='moda',
    # )
    #
    # cap = cv2.VideoCapture(os.path.join(_PROJECT_ROOT, 'video', 'norm.mp4'))
    #
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # inference_times = []
    #
    # writer = PredictionVideoWriter(
    #     save_path=os.path.join(_PROJECT_ROOT, 'output', 'processed_video2.mp4'),
    #     fps=fps,
    #     codec='mp4v',
    # )
    #
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     result = predictor.smooth_prediction(frame)
    #
    #     probs_dict = predictor.probs_to_dict(result['smooth_probs'])
    #     pred_class = result['smooth_class']
    #
    #     annotated = writer.write_frame(frame, probs_dict, pred_class)
    #
    #     inf_ms = result['inference_time_ms']
    #     inf_fps = result['inference_fps']
    #     inference_times.append(inf_ms)
    #     text = f'Inference: {inf_ms:.1f} ms | {inf_fps:.1f} FPS'
    #     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    #     x = annotated.shape[1] - tw - 15
    #     y = 30
    #     cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #
    #     cv2.imshow('prediction', annotated)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #
    # cap.release()
    # writer.release()
    #
    # if inference_times:
    #     avg_ms = sum(inference_times) / len(inference_times)
    #     avg_fps = 1000.0 / avg_ms
    #     print(f'Inference: {avg_ms:.1f} ms/frame (avg) | {avg_fps:.1f} FPS')

    # Минимум — только веса
    predictor = EmbeddingPredictor(weights_path="../weights/EmbeddingModel/final_inference.pth")

    # С классификацией по центроидам
    centroids = np.load("../runs/2026-04-28_22-00-31/embeddings/EmbeddingModel/class_centroids.npy")  # shape: (n_classes, embedding_dim), L2-нормированы
    class_names = ["metal", "stone", "wood", "trash"]

    predictor = EmbeddingPredictor(
        weights_path="../weights/EmbeddingModel/final_inference.pth",
        class_centroids=centroids,
        class_names=class_names,
        smooth_window=5,  # скользящее окно по 5 кадрам
    )

    frame = cv2.imread("../metric_learning_data/val/trash/test_task_arm_false_1-2025_02_14_12_41_48-coco_1.0_0000000007_0.jpg")  # BGR, как из OpenCV / RTSP

    #frame = frame.resize((518, 518))
    result = predictor.predict_with_centroids(frame)

    print(result["pred_class"])  # 'class_a'
    print(result["pred_idx"])  # 0
    print(result["similarities"])  # tensor([0.92, 0.31, 0.18]) — косинусные близости
    print(result["similarities_dict"])  # {'class_a': 0.92, 'class_b': 0.31, 'class_c': 0.18}

    result = predictor.predict_top_k(frame, k=5)

    for entry in result['top_k']:
        print(f"#{entry['rank']:>1}  {entry['pred_class']:<20}  sim={entry['similarity']:.4f}")
