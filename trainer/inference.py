import time
import cv2
import torch
import torch.nn.functional as F

from collections import deque, Counter
from utils.drawer import PredictionVideoWriter
from utils.singeleton_config import ConfigReader
from backbone import FeatureExtractor

cfg = ConfigReader()


class Predictor:
    def __init__(
            self,
            weights_path: str,
            window_size: int = 10,
            class_names: list[str] | None = None,
            smooth_mode: str = 'moda',) -> None:

        self.model_type = cfg.get('MODEL', 'extractor')
        self.model_size = cfg.get('MODEL', 'variant')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.prediction_window = deque(maxlen=window_size)

        self.class_names = class_names or []
        self.smooth_mode = smooth_mode

        if self.smooth_mode not in ('moda', 'mean'):
            raise ValueError(f"Неизвестный smooth_mode: {self.smooth_mode}. Используй 'moda' или 'mean'")

        fe = FeatureExtractor()
        self.model = fe.get_model(
            model_name=self.model_type,
            size=self.model_size,
            input_dim=3,
            output_dim=len(self.class_names),
            clf_mode=True
        )

        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        input_size = cfg.get('MODEL', 'input_size')
        self.input_w = input_size[0]
        self.input_h = input_size[1]

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def prepare_frame(self, frame):
        if frame is None:
            raise ValueError("Кадр пустой")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (self.input_w, self.input_h))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor = tensor.div_(255.0)
        tensor = tensor.sub_(self.mean).div_(self.std)
        return tensor

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
            "probs": probs,
            "pred_idx": pred_idx,
            "pred_class": self.class_names[pred_idx],
            "smooth_probs": smooth_probs,
            "smooth_idx": smooth_idx,
            "smooth_class": self.class_names[smooth_idx],
            "inference_time_ms": inference_time_ms,
            "inference_fps": inference_fps,
        }

    def probs_to_dict(self, probs: torch.Tensor) -> dict[str, float]:
        probs = probs.detach().cpu().tolist()
        return {cls_name: float(prob) for cls_name, prob in zip(self.class_names, probs)}


if __name__ == "__main__":
    predictor = Predictor(weights_path='../weights/DINOv2Extractor/best_checkpoint.pth', window_size=30,
                          class_names=['bad', 'good', 'normal'],
                          smooth_mode='moda')

    cap = cv2.VideoCapture('../video/norm.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)
    inference_times = []

    video_writer = writer = PredictionVideoWriter(
        save_path='../output/processed_video2.mp4',
        fps=fps,
        codec="mp4v")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = predictor.smooth_prediction(frame)

        probs_dict = predictor.probs_to_dict(result["smooth_probs"])
        pred_class = result["smooth_class"]

        annotated = writer.write_frame(frame, probs_dict, pred_class)

        inf_ms = result["inference_time_ms"]
        inf_fps = result["inference_fps"]
        inference_times.append(inf_ms)
        text = f"Inference: {inf_ms:.1f} ms | {inf_fps:.1f} FPS"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = annotated.shape[1] - tw - 15
        y = 30
        cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("prediction", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    writer.release()

    if inference_times:
        avg_ms = sum(inference_times) / len(inference_times)
        avg_fps = 1000.0 / avg_ms
        print(f"Inference: {avg_ms:.1f} ms/frame (avg) | {avg_fps:.1f} FPS")
