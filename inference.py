import cv2
import torch
import torch.nn.functional as F

from cnn_backbone.feature_extractor import DeiT_extractor_small

cls_names = {'bad': 0, 'norm': 1, 'good': 2}
idx_to_class = {v: k for k, v in cls_names.items()}

# Цвет рамки/текста под каждый класс (BGR)
CLS_COLORS = {
    'bad': (0, 0, 255),  # красный
    'norm': (0, 165, 255),  # оранжевый
    'good': (0, 255, 0),  # зелёный
}

resize_shape = (224, 224)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DeiT_extractor_small(input_dim=3, output_dim=3, clf_mode=True)
checkpoint = torch.load('weights/deit/best_model.pth', map_location=device)
model.load_state_dict(
    checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
)
model.to(device)
model.eval()


def prepare_frame(frame):
    if frame is None:
        raise ValueError("Кадр пустой!")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, resize_shape)  # (224, 224, 3)
    tensor = torch.from_numpy(resized).permute(2, 0, 1)  # (3, 224, 224)
    tensor = tensor.unsqueeze(0).float().div_(255.0)  # (1, 3, 224, 224), [0..1]

    # ImageNet normalization — должна совпадать с transforms.Normalize при обучении
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    tensor = tensor.to(device)
    tensor.sub_(mean).div_(std)

    return tensor


def draw_predictions(frame, pred_class: str, probs: dict) -> None:
    """
    Рисует прямо на frame (inplace):
      - цветную рамку по периметру
      - имя предсказанного класса + уверенность (топ) — крупно, вверху
      - бары вероятностей для всех классов — внизу слева
    """
    h, w = frame.shape[:2]
    color = CLS_COLORS[pred_class]

    # ── рамка по периметру ──────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness=4)

    # ── заголовок: класс + уверенность ─────────────────────────────────
    top_conf = probs[pred_class]
    top_label = f"{pred_class.upper()}  {top_conf * 100:.1f}%"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.1
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(top_label, font, font_scale, thickness)
    pad = 8

    # фон под текст
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2 + baseline),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, top_label,
                (8 + pad, 8 + th + pad),
                font, font_scale, color, thickness, cv2.LINE_AA)

    # ── бары вероятностей всех классов ─────────────────────────────────
    bar_x = 12
    bar_h = 20
    bar_max_w = 200
    bar_gap = 30  # шаг между строками
    start_y = h - (len(probs) * bar_gap) - 16

    small_font = cv2.FONT_HERSHEY_SIMPLEX
    small_scale = 0.55
    small_thick = 1

    for i, (cls_name, prob) in enumerate(probs.items()):
        y = start_y + i * bar_gap
        bar_w = int(prob * bar_max_w)
        clr = CLS_COLORS[cls_name]
        is_top = cls_name == pred_class

        # фоновый бар (тёмный)
        cv2.rectangle(frame,
                      (bar_x, y),
                      (bar_x + bar_max_w, y + bar_h),
                      (40, 40, 40), cv2.FILLED)

        # заполненный бар
        if bar_w > 0:
            cv2.rectangle(frame,
                          (bar_x, y),
                          (bar_x + bar_w, y + bar_h),
                          clr, cv2.FILLED)

        # подсветка активного класса
        if is_top:
            cv2.rectangle(frame,
                          (bar_x - 2, y - 2),
                          (bar_x + bar_max_w + 2, y + bar_h + 2),
                          clr, thickness=2)

        # метка: "good  84.3%"
        label = f"{cls_name:<14} {prob * 100:.1f}%"
        cv2.putText(frame, label,
                    (bar_x + bar_max_w + 8, y + bar_h - 4),
                    small_font, small_scale,
                    clr if is_top else (200, 200, 200),
                    small_thick, cv2.LINE_AA)


video_path = 'dataset/video/norm.mp4'
output_path = 'dataset/video/norm_result.mp4'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps, (fw, fh)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tf_frame = prepare_frame(frame)

    with torch.no_grad():
        raw_pred = model(tf_frame)
        prob_pred = F.softmax(raw_pred, dim=1)[0]

    pred_idx = torch.argmax(prob_pred).item()
    pred_class = idx_to_class[pred_idx]

    probs = {idx_to_class[i]: prob_pred[i].item() for i in range(len(idx_to_class))}

    draw_predictions(frame, pred_class, probs)

    writer.write(frame)
    cv2.imshow('Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
