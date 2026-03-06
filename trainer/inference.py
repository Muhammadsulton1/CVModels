from collections import deque
import cv2
import torch
import torch.nn.functional as F

from utils.singeleton_config import ConfigReader
from trainer.backbone import FeatureExtractor
from utils.drawer import PredictionVideoWriter

cfg = ConfigReader()


class Predictor:
    def __init__(self, weights_path: str, window_size: int = 10, class_names: list[str] | None = None) -> None:
        model_type = cfg.get('MODEL', 'extractor')
        model_size = cfg.get('MODEL', 'variant')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.prediction_window = deque(maxlen=window_size)

        self.class_names = class_names

        self.model = FeatureExtractor.get_model(
            model_type,
            model_size,
            input_dim=3,
            output_dim=len(self.class_names),
            clf_mode=True)

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

    @torch.no_grad()
    def smooth_prediction(self, frame):
        probs, pred_idx = self.predict(frame)

        self.prediction_window.append(probs.detach().cpu())

        stacked = torch.stack(list(self.prediction_window), dim=0)
        smooth_probs = stacked.mean(dim=0)
        smooth_idx = int(torch.argmax(smooth_probs).item())

        return {
            "probs": probs,
            "pred_idx": pred_idx,
            "pred_class": self.class_names[pred_idx],
            "smooth_probs": smooth_probs,
            "smooth_idx": smooth_idx,
            "smooth_class": self.class_names[smooth_idx],
        }

    def probs_to_dict(self, probs: torch.Tensor) -> dict[str, float]:
        probs = probs.detach().cpu().tolist()
        return {cls_name: float(prob) for cls_name, prob in zip(self.class_names, probs)}


if __name__ == "__main__":
    predictor = Predictor(weights_path='./weights/weights.pth', window_size=10, class_names=['good', 'normal', 'bad'])

    cap = cv2.VideoCapture('video/bad.mp4')

    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = writer = PredictionVideoWriter(
        save_path='output/processed_video1.mp4',
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

        cv2.imshow("prediction", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cap.release()
        writer.release()