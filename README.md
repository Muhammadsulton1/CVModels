# CVModels — фреймворк для обучения классификаторов изображений

Проект для обучения и инференса моделей компьютерного зрения. Поддерживает множество архитектур (ResNet, ViT, DeiT, DINOv2, EfficientNet и др.), гибкую конфигурацию через YAML и универсальную работу с различными форматами датасетов.

---

## Содержание

1. [Структура проекта](#структура-проекта)
2. [Установка](#установка)
3. [Быстрый старт](#быстрый-старт)
4. [Конфигурация](#конфигурация)
5. [Обучение](#обучение)
6. [Работа с датасетами](#работа-с-датасетами)
7. [Инференс](#инференс)
8. [Архитектуры моделей](#архитектуры-моделей)
9. [API и расширение](#api-и-расширение)

---

## Структура проекта

```
CVModels/
├── config/
│   └── train_conf.yaml      # Главный конфиг обучения
├── cnn_backbone/
│   ├── feature_extractor.py  # Архитектуры моделей
│   ├── trainer.py            # Логика обучения
│   ├── model_train.py        # Скрипт запуска обучения
│   └── ConvNext.py           # ConvNeXt (если используется)
├── utils/
│   ├── singeleton_config.py  # Синглтон для чтения конфига
│   └── logger.py             # Логирование
├── inference.py              # Инференс на видео
├── weights/                  # Сохранённые чекпоинты (создаётся при обучении)
│   └── {ModelName}/
│       └── best_checkpoint.pth
└── README.md
```

---

## Установка

### Зависимости

```bash
pip install torch torchvision
pip install omegaconf scikit-learn tqdm tensorboard
pip install timm transformers  # для ViT, DeiT, DINOv2
```

### Проверка

```bash
cd CVModels
python -c "from cnn_backbone.trainer import Trainer; from utils.singeleton_config import ConfigReader; print('OK')"
```

---

## Быстрый старт

### 1. Подготовка данных

Структура для **ImageFolder** (рекомендуется для начала):

```
my_dataset/
├── class_a/
│   ├── img1.jpg
│   └── img2.jpg
├── class_b/
│   └── ...
└── class_c/
    └── ...
```

### 2. Настройка конфига

Отредактируйте `config/train_conf.yaml`:

```yaml
MODEL:
  input_dim: 3
  output_dim: 3          # число классов
  use_clf: true
  input_size: [224, 224]

DATASET:
  num_classes: null       # null = автоопределение
```

### 3. Запуск обучения

```python
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from trainer.feature_extractor import DINOv2Extractor
from trainer.trainer import Trainer

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root='my_dataset', transform=transform)
model = DINOv2Extractor(3, 3, 'small_reg', clf_mode=True)
criterion = CrossEntropyLoss(reduction='mean')

trainer = Trainer(model, criterion, dataset)
trainer.fit()
```

Или запустите готовый скрипт:

```bash
python trainer/model_train.py
```

Чекпоинты сохраняются в `weights/{ИмяМодели}/best_checkpoint.pth`.

---

## Конфигурация

Весь проект управляется одним файлом `config/train_conf.yaml`. Конфиг загружается через синглтон `ConfigReader` — один экземпляр на всё приложение.

### Секция MODEL

| Параметр    | Описание                          | Пример      |
|------------|------------------------------------|-------------|
| extractor  | Имя модели для FeatureExtractor    | deit_small_patch16 |
| input_dim  | Число каналов входа (1 или 3)      | 3           |
| output_dim | Число классов                     | 3           |
| use_clf    | Режим классификатора               | true        |
| input_size | Размер входа [H, W]               | [224, 224]  |

### Секция DATASET

| Параметр           | Описание | Пример |
|--------------------|----------|--------|
| num_classes        | Число классов (`null` = авто) | null |
| targets_attrs      | Атрибуты датасета с метками (по приоритету) | [targets, labels, y] |
| class_mapping_attr | Атрибут с маппингом имён→индексы | back_names, cls_names, null |
| label_key          | Ключ в dict-элементе для метки | quality_label, label, null |

### Секция TRAINER

| Параметр         | Описание | Пример |
|------------------|----------|--------|
| optimizer_type   | Оптимизатор | AdamW, RAdam, Adam, SGD, RMSprop, NAdam, Adamax |
| scheduler_type   | Шедулер LR | cosine, cosine_with_warmup, steplr, multisteplr, reduceonplateau, exponential, none |
| num_epochs       | Число эпох | 50 |
| warmup_epochs    | Эпохи warmup (для cosine_with_warmup) | 5 |
| lr               | Learning rate | 1e-4 |
| weight_decay     | L2-регуляризация | 0.05 |
| batch_size       | Размер батча | 32 |
| num_workers      | Воркеры DataLoader | 8 |
| train_proc       | Доля train (0.0–1.0) | 0.75 |
| split_strategy   | Разбиение train/val | random, sequential, stratified |
| use_class_weights | Веса классов для дисбаланса | true |
| train_mode       | Режим обучения | scratch, finetune, resume |
| load_checkpoint  | Путь к чекпоинту (для finetune/resume) | weights/best.pth |

### Секция OPTIMIZER

| Параметр | Описание | Оптимизаторы |
|----------|----------|--------------|
| betas    | [β1, β2] для Adam-семейства | [0.9, 0.999] |
| eps      | Эпсилон | 1e-8 |
| momentum | Momentum для SGD/RMSprop | 0.9 |
| nesterov | Nesterov для SGD | true |

### Секция SCHEDULER

| Параметр | Описание | Шедулеры |
|----------|----------|----------|
| step_size | Шаг для StepLR | 10 |
| gamma     | Множитель LR | 0.1 |
| milestones | Эпохи для MultiStepLR | [30, 60, 90] |
| plateau_mode, plateau_factor, plateau_patience, plateau_min_lr | ReduceLROnPlateau | — |
| exp_gamma | Gamma для ExponentialLR | 0.95 |
| eta_min   | Минимальный LR для Cosine | 0 |
| warmup_start_factor | Начальный множитель для warmup | 0.01 |

### Секция EARLY_STOPPING

| Параметр    | Описание | Пример |
|-------------|----------|--------|
| patience    | Эпохи без улучшения до остановки | 10 |
| min_delta   | Минимальное улучшение val_loss | 0.01 |
| n_best_nets | Число лучших чекпоинтов для Model Soup | 5 |
| soup_strategy | Стратегия усреднения | avg |

### Секция METRICS

| Параметр    | Описание | Пример |
|-------------|----------|--------|
| enabled    | Включить метрики на валидации | true |
| list       | Список метрик | accuracy, f1_macro, confusion_matrix |
| class_names | Имена классов для confusion matrix | ["bad", "norm", "good"] |

Доступные метрики: `accuracy`, `precision_macro`, `precision_micro`, `precision_weighted`, `recall_macro`, `recall_micro`, `recall_weighted`, `f1_macro`, `f1_micro`, `f1_weighted`, `confusion_matrix`.

---

## Обучение

### Режимы обучения (train_mode)

- **scratch** — обучение с нуля
- **finetune** — загрузка чекпоинта, дообучение (оптимизатор сбрасывается)
- **resume** — продолжение обучения (оптимизатор и эпоха восстанавливаются)

### Сохранение чекпоинтов

Чекпоинты сохраняются в:

```
weights/{ИмяКлассаМодели}/best_checkpoint.pth
```

Например: `weights/DINOv2Extractor/best_checkpoint.pth`.

### Model Soup

После обучения применяется усреднение весов лучших чекпоинтов (Model Soup). Количество чекпоинтов задаётся в `EARLY_STOPPING.n_best_nets`.

### TensorBoard

```bash
tensorboard --logdir runs
```

### Логи

Логи пишутся в `training.log` и в консоль.

---

## Работа с датасетами

Тренажёр поддерживает разные форматы датасетов.

### 1. ImageFolder (torchvision)

Стандартная структура папок. Метки берутся из `dataset.targets` или `dataset.classes` / `dataset.class_to_idx`.

```yaml
dataset:
  targets_attrs: [targets, labels, y]
  # class_mapping_attr и label_key не нужны
```

### 2. Кастомный датасет с атрибутом меток

Если у датасета есть атрибут `my_labels`:

```yaml
dataset:
  targets_attrs: [my_labels]
```

### 3. Датасет с `data` (список dict-элементов)

Если датасет хранит `data` — список словарей, и метка в ключе `quality_label`:

```yaml
dataset:
  targets_attrs: [targets, labels, y]
  # Если метки в data[i]['quality_label']:
  label_key: quality_label
  # Если есть маппинг имён классов в индексы:
  class_mapping_attr: back_names
```

Для других имён атрибутов:

```yaml
dataset:
  label_key: my_label
  class_mapping_attr: cls_names
```

### 4. Tuple/List элементы

Если `__getitem__` возвращает `(image, label)` — метка берётся автоматически как второй элемент.

### Автоопределение числа классов

- `DATASET.num_classes: null` — из датасета (`classes`, `class_to_idx`) или из меток
- `DATASET.num_classes: 5` — явно задано

### Стратификация

При `split_strategy: stratified` разбиение сохраняет пропорции классов. Требуются метки для всех сэмплов.

---

## Инференс

Скрипт `inference.py` выполняет inference на видео.

### Настройка

1. Укажите путь к чекпоинту и модель.
2. Задайте `cls_names` и `idx_to_class` в соответствии с обучением.
3. Укажите `resize_shape` как в `MODEL.input_size`.

### Запуск

```bash
python inference.py
```

В коде задаются:

```python
video_path = 'dataset/video/norm.mp4'
output_path = 'dataset/video/norm_result.mp4'
```

Используйте `q` для выхода из превью.

---

## Архитектуры моделей

### Доступные через FeatureExtractor

| Ключ | Класс |
|------|-------|
| resnet18 | Resnet18Extractor |
| resnet34 | Resnet34Extractor |
| resnet50 | Resnet50Extractor |
| mobilenetv2 | MobileNetV2_extractor |
| shufflenet_v2_x0_5 | ShuffleNetV2_extractor_x05 |
| shufflenet_v2_x1_0 | ShuffleNetV2_extractor_x10 |
| shufflenet_v2_x1_5 | ShuffleNetV2_extractor_x15 |
| vit_b16 | Vit_b16_extractor |
| vit_b32 | Vit_b32_extractor |
| mobileNetV3_small | MobileNetV3Extractor_small |
| mobileNetV3_large | MobileNetV3Extractor_large |
| regnet_y_400mf | RegnetY400mf_Extractor |
| regnet_y_800mf | RegnetY800mf_Extractor |
| efficientnet_b0 | Efficientnet_b0_extractor |
| efficientnet_b1 | Efficientnet_b1_extractor |
| efficientnet_b2 | Efficientnet_b2_extractor |
| inception_v3 | InceptionV3_extractor |
| deit_tiny_patch16 | DeiT_extractor |
| deit_small_patch16 | DeiT_extractor_small |

### Прямое использование

```python
from trainer.feature_extractor import (
    DINOv2Extractor,  # DINOv2 (small_reg, base, large)
    DeiT_extractor_small,
    Resnet50Extractor,
    Efficientnet_b0_extractor,
    # ...
)
```

### Регистрация своей модели

```python
from trainer.feature_extractor import FeatureExtractor

fe = FeatureExtractor()
fe.register_model('my_model', MyModelClass)
model = fe.extract_features('my_model', input_dim=3, output_dim=3, clf_mode=True)
```

---

## API и расширение

### ConfigReader

```python
from utils.singeleton_config import ConfigReader

cfg = ConfigReader()
value = cfg.get('TRAINER', 'batch_size', default=32)
section = cfg.get_section('MODEL')
```

### Trainer

```python
trainer = Trainer(model, criterion, dataset)
train_loader, val_loader = trainer.prepare_data()
optimizer = trainer.init_optimizer(model, lr=1e-4)
scheduler = trainer.init_scheduler(optimizer, lr, num_epochs, scheduler_type)
trainer.fit()
```

### Требования к модели

- `forward(x)` возвращает тензор `(batch_size, num_classes)` в режиме классификации
- Поддержка `x.dim() == 5` (batch of sequences) — опционально

### Требования к датасету

- `__len__()` — число сэмплов
- `__getitem__(i)` — `(image_tensor, label_int)` или совместимый формат
- Метки для стратификации и весов классов — через `targets_attrs`, `label_key`, `class_mapping_attr`

---

## Частые проблемы

**FileNotFoundError: Config not found**  
Запускайте скрипты из корня проекта `CVModels` или убедитесь, что `config/train_conf.yaml` существует.

**Ошибка при stratified split**  
Слишком мало сэмплов в каком-то классе. Используйте `split_strategy: random` или увеличьте датасет.

**Не находятся метки**  
Проверьте `targets_attrs`, `label_key`, `class_mapping_attr` в конфиге под формат вашего датасета.

**Чекпоинт не загружается**  
Укажите полный путь в `load_checkpoint` и проверьте, что модель совпадает с сохранённой.

---

## Лицензия

MIT (если не указано иное).
