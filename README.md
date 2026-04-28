# CVModels

Репозиторий для **обучения моделей компьютерного зрения** на ваших данных: задача **классификации изображений** и отдельно **метрическое обучение эмбеддингов** (поиск/кластеры по косинусной близости). В качестве бэкенов используются готовые извлекатели признаков (в т.ч. **timm**, **transformers**/DINO-подобные модели через `models_src`), настройки — в **`config/train_conf.yaml`**.

---

## Зачем это нужно

- **Нужно отнести кадры к одному из классов** (например, тип отходов, дефект, сцена) — обучайте классификатор (`train_src/classification_train.py`).
- **Нужны компактные векторные представления**, чтобы сравнивать образцы, строить поиск похожих, кластеры, центроиды классов — используйте эмбеддинги (`train_src/embedding_train.py`).
- Единый **YAML‑конфиг**, логирование, TensorBoard, ранняя остановка, опционально **Model Soup** (усреднение последних лучших весов).

**С чего начать:** раздел **[Примеры: как запустить по шагам](#примеры-как-запустить-по-шагам)** (команды `cd`, `pip`, запуск классификации и эмбеддингов, TensorBoard, инференс).

---

## Структура проекта

```
CVModels/
├── config/
│   └── train_conf.yaml       # Основные гиперпараметры, лосс для эмбеддингов, метрики
├── models_src/
│   ├── clf_model.py          # Голова классификации + backbone из ModelFactory
│   ├── emb_model.py           # Выход эмбеддингов + backbone
│   ├── get_feature.py         # Фабрика моделей (timm/transformers/DINO и др.)
│   └── encoder.py
├── train_src/
│   ├── trainer.py             # Общая логика: оптимайзеры, LR, EarlyStopping, soup, графики loss
│   ├── classification_train.py # Запуск классификации
│   ├── embedding_train.py      # Запуск эмбеддингов / metric learning
│   └── inference.py           # Инференс с видео/кадров (Predictor / EmbeddingPredictor)
├── emb_loss/
│   └── losses.py              # SupCon, ArcLoss, Center*, комбинации
├── utils/                     # логгер, конфиг-синглтон, вспомогательные утилиты
├── metric_learning_data/      # Пример имени каталога с данными (задаётся в DATASET.root)
├── weights/                   # Чекпоинты после обучения (создаются автоматически)
└── runs/                      # Логи TensorBoard и сохранённые графики запусков
```

### Свои классы моделей (`clf_model.py`, `emb_model.py`)

Файлы **`models_src/clf_model.py`** (`ModelClassification`) и **`models_src/emb_model.py`** (`EmbeddingModel`) по сути **эталонные примеры**: как взять `backbone` из **`ModelFactory`** (конфиг `MODEL` в `train_conf.yaml`), навесить голову и отдать модель тренеру.

| Задача | Точка входа | Что должно быть на выходе `forward` |
|--------|-------------------|-------------------------------------|
| Классификация | `classification_train.py` → `TrainerClassification` | Логиты `(batch, число классов)` |
| Эмбеддинги | `embedding_train.py` → `TrainerEmbeddings` | Обычно `(batch, D)` при **L2-нормировке по строкам** (как у референса); для лоссов в `emb_loss` задайте **`embedding_dim` у модели**. |

Чтобы использовать **свою** архитектуру головы: скопируйте один из модулей (или только класс), поправьте слои под себя и в **`classification_train.py` / `embedding_train.py`** замените импорт и конструктор — остальной конвейер (данные, лосс для эмбеддингов, чекпоинты, тренер) тот же. См. также блок `if __name__ == '__main__'` в этих файлах — там минимальный «дымовой» запуск без тренера.

---

## Установка

Из корня репозитория:

```bash
pip install -r requirements.txt
```

Рекомендуется использовать **CUDA** для обучения; CPU возможен, но медленнее.

---

## Примеры: как запустить по шагам

Ниже — минимальный сценарий «поставил зависимости → указал данные → запустил обучение».

### 1. Перейти в корень репозитория

Рабочая директория должна содержать `config/train_conf.yaml` (иначе `ConfigReader` не найдёт конфиг).

```bash
cd /path/to/CVModels          # Linux / macOS
```

```powershell
cd C:\path\to\CVModels        # Windows PowerShell или CMD
```

Опционально — виртуальное окружение:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Подготовить данные

См. [Подготовка данных](#подготовка-данных-imagefolder). Кратко: каталог `DATASET.root` с подпапками `train/` и `val/`, внутри — папки классов с изображениями.

### 4. Правки в `config/train_conf.yaml` (минимум)

Замените путь к данным и при необходимости длительность обучения:

```yaml
DATASET:
  root: metric_learning_data    # или my_data — папка рядом с проектом / абсолютный путь

TRAINER:
  train_mode: scratch           # scratch | finetune | resume
  num_epochs: 50
  batch_size: 16
  lr: 1.0e-4
```

Если путь **относительный**, он трактуется **от корня проекта** (папка, где лежит `config/`).

### 5. Запуск обучения

**Классификация** (кросс-энтропия, метрики из секции `METRICS`):

```bash
python train_src/classification_train.py
```

**Эмбеддинги / metric learning** (лосс из `EMBEDDING_LOSS`):

```bash
python train_src/embedding_train.py
```

На **Windows**, если команда `python` не найдена, часто срабатывает лаунчер:

```powershell
py train_src/classification_train.py
py train_src/embedding_train.py
```

Во время и после обучения смотрите консоль и при необходимости **TensorBoard**:

```bash
tensorboard --logdir runs
```

Откройте в браузере адрес, который выведет TensorBoard (обычно `http://localhost:6006`).

### 6. Где лежат результаты

| Что | Где |
|-----|-----|
| Логи TensorBoard, `loss_curves.png` | `runs/ГГГГ-ММ-ДД_ЧЧ-ММ-СС/` |
| Лучший чекпоинт по early stopping | `weights/<ИмяМодели>/best_checkpoint.pth` |
| Веса после Model Soup (для инференса «как в конце fit») | `weights/<ИмяМодели>/final_inference.pth` |
| Для эмбеддингов: центроиды / картинка кластеров (если включено) | внутри того же `runs/.../embeddings/...` |

Имя папки под `weights/` совпадает с классом модели, например `ModelClassification` или `EmbeddingModel`.

### 7. Продолжить или донастроить

В `TRAINER` задайте:

```yaml
train_mode: resume              # или finetune
load_checkpoint: weights/EmbeddingModel/best_checkpoint.pth   # пример; путь подставьте свой
```

Путь к чекпоинту — **от корня проекта** или абсолютный (см. реализацию загрузки в `train_src/trainer.py`).

### 8. Инференс (пример с видео)

В репозитории есть рабочий пример в конце **`train_src/inference.py`**: классификация по кадрам видео с плавностью по окну. Перед запуском отредактируйте в файле:

- путь к весам `weights_path` (часто удобнее `final_inference.pth` или `best_checkpoint.pth` из шага 6);
- список `class_names` **в том же порядке**, что и при обучении;
- пути к входному видео и выходному файлу.

Запуск из корня проекта:

```bash
python train_src/inference.py
```

---

## Подготовка данных (ImageFolder)

Ожидается раздельное **train** и **val** (как два `ImageFolder`):

```
<DATASET.root>/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── val/
    ├── class_a/
    ├── class_b/
    └── ...
```

Имена подпапок и **порядок классов** должны быть **одинаковыми** в `train` и `val` (иначе будет ошибка проверки `class_to_idx`). В конфиге указываете путь `DATASET.root` (относительно корня проекта или абсолютный).

---

## Конфигурация (`config/train_conf.yaml`)

Центральный файл: его читает **синглтон** `utils/singeleton_config.ConfigReader` — правки затрагивают все точки входа.

| Секция | Назначение |
|--------|------------|
| **MODEL** | Backbone (`dinov2` и др.), размер входа `img_size`, `freeze_backbone`, для классификации задаётся `output_dims` (число классов; скрипт может перезаписать из данных). |
| **DATASET** | `root` к данным, `num_classes` — можно зафиксировать или уточнить из данных. |
| **TRAINER** | Эпохи, LR, `batch_size`, `optimizer_type`, `scheduler_type`, `train_mode` (`scratch` / `finetune` / `resume`), `load_checkpoint` при донастройке/продолжении. |
| **EARLY_STOPPING** | `monitor` (`val_loss` или имя метрики из валидации, например `recall_at_1`), `mode` (`min` / `max`), `patience`, `n_best_nets`, Model Soup (`soup_strategy: avg`). |
| **METRICS** | Для классификации: список метрик, `class_names` для матрицы ошибок. Скрипт классификации может синхронизировать имена с папками `train`. |
| **EMBEDDING_LOSS** | Для эмбеддингов: `name` (`supcon`, `arc`, `center`, ...), параметры температуры/Arc/Center и **balanced batches** для SupCon (`classes_per_batch`, `samples_per_class`). |
| **EMBEDDING_METRICS** | Включение доп.метрик по валидации, сохранение центроидов, отрисовка кластеров после обучения. |

---

## Обучение: классификация

Подробные команды запуска, правки конфига и пути к весам — в разделе **[Примеры: как запустить по шагам](#примеры-как-запустить-по-шагам)**.

Кратко:

```bash
python train_src/classification_train.py
```

Используется `TrainerClassification`: кросс-энтропия; метрики из секции `METRICS`, если включены и задан список.

---

## Обучение: эмбеддинги (metric learning)

Полный порядок действий такой же, как в **[Примеры: как запустить по шагам](#примеры-как-запустить-по-шагам)**; ниже отличие только командой входа и секциями конфига `EMBEDDING_*`.

Подходит для **SupCon**/Arc**/Center*** и смесей через `emb_loss`:

```bash
python train_src/embedding_train.py
```

- В конфиге настраивается **`EMBEDDING_LOSS`**, при включённом `balanced_batches` батчи собираются с несколькими примерами на класс для контраста.
- Валидация может считать recall@k и др.; после `fit()` в каталог запуска в `runs/.../` сохраняются **кривая потерь** (`loss_curves.png`), при настройках — **кластеры/центроиды** для эмбеддингов.

---

## Чекпоинты

После успешной настройки путей веса пишутся в:

```
weights/<ИмяКлассаМодели>/best_checkpoint.pth      # Последнее лучшее состояние по раннему мониторингу
weights/<ИмяКлассаМодели>/final_inference.pth      # После возможного Model Soup — совпадает с весами в памяти в конце fit()
```

- Для типичной донастройки и «best по метрике» используйте **`best_checkpoint.pth`**.
- Если после обучения применился **усредняющий Model Soup**, для инференса в точности такой же модели используйте **`final_inference.pth`**.

Структура файла включает `model_state_dict`, при наличии — `criterion_state_dict` и `optimizer_state_dict`.

---

## Model Soup

При `EARLY_STOPPING.soup_strategy: avg` и ненулевом `n_best_nets` после цикла эпох веса **усредняются** по сохранённым лучшим снимкам. Итоговые параметры сохраняются в **`final_inference.pth`** (см. выше).

---

## TensorBoard и логи

```bash
tensorboard --logdir runs
```

Каждый запуск создаёт подпапку `runs/ГГГГ-ММ-ДД_ЧЧ-ММ-СС` с трассами потерь, LR и метрик.

---

## Инференс

Модуль **`train_src/inference.py`** задаёт классы **`Predictor`** (классификация по softmax) и **`EmbeddingPredictor`** (эмбеддинги; опционально сравнение с заранее заданными центроидами).

**Минимальный сценарий** (пути к весам и видео поправьте под себя):

1. В `if __name__ == '__main__':` в конце файла укажите `weights_path`, список **`class_names`** в том же порядке, что при обучении, и путь к видео `video/...`.
2. Из корня проекта: `python train_src/inference.py`
3. Выход из превью кадра: клавиша **`q`**.

Детали и пример блока с `cv2.VideoCapture` — в коде файла; параметры препроцесса должны соответствовать **`MODEL.img_size`** из `config/train_conf.yaml`.

---

## Частые проблемы

| Проблема | Что проверить |
|----------|----------------|
| `Config not found` | Запуск из корня `CVModels`, наличие `config/train_conf.yaml`. |
| Разные `class_to_idx` у train/val | Одинаковый набор и имена папок классов в `train/` и `val/`. |
| `output_dims` / число классов | Для классификации задайте `MODEL.output_dims` или доверьте скрипту после `ImageFolder`. |
| Память на валидации эмбеддингов | Очень большой `val` увеличивает время/память — уменьшите выборку или `batch_size`. |
| Несовпадение чекпоинта и «финальной» модели | Используйте **`final_inference.pth`** после soup; **`best_checkpoint.pth`** — лучший снимок в процессе обучения. |

---
