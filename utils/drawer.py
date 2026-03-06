from pathlib import Path
import colorsys
import cv2
import numpy as np


class PredictionVideoWriter:
    def __init__(
        self,
        save_path: str,
        fps: float = 25.0,
        codec: str = "mp4v",
        class_colors: dict | None = None,
        bar_max_w: int = 220,
        border_thickness: int = 4,
        top_k_first: bool = True,
    ):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.codec = codec
        self.class_colors = class_colors or {}
        self.bar_max_w = bar_max_w
        self.border_thickness = border_thickness
        self.top_k_first = top_k_first

        self.writer = None
        self.frame_size = None

    def _make_writer(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(str(self.save_path), fourcc, self.fps, self.frame_size)

        if not self.writer.isOpened():
            raise RuntimeError(f"Не удалось открыть VideoWriter для {self.save_path}")

    @staticmethod
    def _generate_color_map(class_names: list[str]) -> dict:
        n = max(len(class_names), 1)
        color_map = {}

        for i, cls_name in enumerate(sorted(class_names)):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            color_map[cls_name] = (int(b * 255), int(g * 255), int(r * 255))  # BGR для OpenCV

        return color_map

    def _ensure_colors(self, probs: dict) -> None:
        missing = [cls_name for cls_name in probs.keys() if cls_name not in self.class_colors]
        if missing:
            auto_map = self._generate_color_map(list(probs.keys()))
            for cls_name in missing:
                self.class_colors[cls_name] = auto_map[cls_name]

    @staticmethod
    def _resolve_pred_class(probs: dict, pred_class: str | None = None) -> str:
        if pred_class is not None:
            return pred_class
        if not probs:
            raise ValueError("probs пустой")
        return max(probs, key=probs.get)

    def draw_predictions(
        self,
        frame: np.ndarray,
        probs: dict[str, float],
        pred_class: str | None = None,
    ) -> np.ndarray:
        if frame is None:
            raise ValueError("frame is None")
        if not probs:
            raise ValueError("probs пустой")

        pred_class = self._resolve_pred_class(probs, pred_class)
        self._ensure_colors(probs)

        out = frame.copy()
        h, w = out.shape[:2]
        color = self.class_colors[pred_class]

        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness=self.border_thickness)

        top_conf = probs[pred_class]
        top_label = f"{pred_class.upper()}  {top_conf * 100:.1f}%"

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.1
        thickness = 2
        pad = 8

        (tw, th), baseline = cv2.getTextSize(top_label, font, font_scale, thickness)

        cv2.rectangle(
            out,
            (8, 8),
            (8 + tw + pad * 2, 8 + th + pad * 2 + baseline),
            (0, 0, 0),
            cv2.FILLED
        )
        cv2.putText(
            out,
            top_label,
            (8 + pad, 8 + th + pad),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        items = list(probs.items())
        if self.top_k_first:
            items = sorted(items, key=lambda x: x[1], reverse=True)

        bar_x = 12
        bar_h = 20
        bar_gap = 30
        start_y = h - (len(items) * bar_gap) - 16

        small_font = cv2.FONT_HERSHEY_SIMPLEX
        small_scale = 0.55
        small_thick = 1

        for i, (cls_name, prob) in enumerate(items):
            y = start_y + i * bar_gap
            bar_w = int(max(0.0, min(1.0, prob)) * self.bar_max_w)
            clr = self.class_colors[cls_name]
            is_top = cls_name == pred_class

            cv2.rectangle(
                out,
                (bar_x, y),
                (bar_x + self.bar_max_w, y + bar_h),
                (40, 40, 40),
                cv2.FILLED
            )

            if bar_w > 0:
                cv2.rectangle(
                    out,
                    (bar_x, y),
                    (bar_x + bar_w, y + bar_h),
                    clr,
                    cv2.FILLED
                )

            if is_top:
                cv2.rectangle(
                    out,
                    (bar_x - 2, y - 2),
                    (bar_x + self.bar_max_w + 2, y + bar_h + 2),
                    clr,
                    thickness=2
                )

            label = f"{cls_name:<14} {prob * 100:.1f}%"
            cv2.putText(
                out,
                label,
                (bar_x + self.bar_max_w + 8, y + bar_h - 4),
                small_font,
                small_scale,
                clr if is_top else (200, 200, 200),
                small_thick,
                cv2.LINE_AA
            )

        return out

    def write_frame(
        self,
        frame: np.ndarray,
        probs: dict[str, float],
        pred_class: str | None = None,
    ) -> np.ndarray:
        if self.writer is None:
            self._make_writer(frame)

        if self.frame_size != (frame.shape[1], frame.shape[0]):
            raise ValueError(
                f"Размер кадра изменился. "
                f"Ожидался {self.frame_size}, получен {(frame.shape[1], frame.shape[0])}"
            )

        annotated = self.draw_predictions(frame, probs, pred_class)
        self.writer.write(annotated)
        return annotated

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
