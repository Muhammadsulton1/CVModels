import os
import sys
import time
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from omegaconf import OmegaConf

from train_src.early_stopping import EarlyStopping
from utils.trainer_utils import (
    _embedding_retrieval_metrics_at_k,
    _plot_embedding_clusters_2d,
)
from utils.logger import setup_logger
from utils.singeleton_config import ConfigReader

logger = setup_logger()

METRIC_FUNCS = {
    'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
    'precision_micro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro', zero_division=0),
    'precision_weighted': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=0),
    'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
    'recall_micro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro', zero_division=0),
    'recall_weighted': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted', zero_division=0),
    'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro', zero_division=0),
    'f1_micro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro', zero_division=0),
    'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted', zero_division=0),
}


class BalancedClassBatchSampler(Sampler[list[int]]):
    """
    Batch-сэмплер с балансировкой классов для metric learning.

    Каждый батч содержит строго ``classes_per_batch`` классов
    и ``samples_per_class`` примеров на класс, что гарантирует
    наличие достаточного числа позитивных пар для SupConLoss / TripletLoss / ArcFace.

    Структура батча:
        [cls_0 × K, cls_1 × K, ..., cls_C × K]  — перемешано перед yield

    Требования:
        - Минимум 2 класса в датасете
        - classes_per_batch ≥ 2, samples_per_class ≥ 2

    Если класс содержит меньше ``samples_per_class`` примеров, сэмплирование
    выполняется с заменой (``replace=True``).

    Аргументы:
        dataset: датасет с атрибутом targets / labels / y или samples.
            Поддерживает Subset (обёртку с атрибутом .indices).
        classes_per_batch (int): число классов в одном батче.
        samples_per_class (int): число примеров на класс в батче.
        batches_per_epoch (int | None): число батчей на эпоху.
            По умолчанию ``len(dataset) // (classes_per_batch * samples_per_class)``.

    Пример:
        sampler = BalancedClassBatchSampler(train_ds, classes_per_batch=8, samples_per_class=4)
        loader  = DataLoader(train_ds, batch_sampler=sampler)
        # Каждый батч: 32 примера, 8 классов × 4 примера
    """

    def __init__(self, dataset, classes_per_batch: int, samples_per_class: int, batches_per_epoch: int | None = None) -> None:
        super().__init__()
        labels = self._labels_from_dataset(dataset)
        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[int(label)].append(idx)

        self.class_to_indices = {c: idxs for c, idxs in class_to_indices.items() if idxs}
        self.classes = sorted(self.class_to_indices)
        if len(self.classes) < 2:
            raise ValueError('BalancedClassBatchSampler needs at least two classes.')

        self.classes_per_batch = min(int(classes_per_batch), len(self.classes))
        self.samples_per_class = int(samples_per_class)
        if self.classes_per_batch < 2 or self.samples_per_class < 2:
            raise ValueError('Use at least 2 classes per batch and 2 samples per class for SupConLoss.')

        batch_size = self.classes_per_batch * self.samples_per_class
        self.batches_per_epoch = batches_per_epoch or max(1, len(labels) // batch_size)

    @staticmethod
    def _labels_from_dataset(dataset) -> list[int]:
        """
        Универсально извлекает метки из датасета.

        Поддерживаемые форматы хранения меток:
            - ``dataset.targets``   — torchvision ImageFolder, MNIST, CIFAR и др.
            - ``dataset.labels``    — кастомные датасеты
            - ``dataset.y``         — sklearn-совместимые датасеты
            - ``dataset.samples``   — список (path, class_idx), ImageFolder.samples

        Поддерживает ``torch.utils.data.Subset`` через разыменование ``dataset.indices``.

        Аргументы:
            dataset: любой датасет PyTorch.

        Возвращает:
            list[int]: список целочисленных меток в том же порядке, что и датасет.

        Исключения:
            ValueError: если не удалось найти метки ни одним из способов.
        """
        indices = getattr(dataset, 'indices', None)
        base = getattr(dataset, 'dataset', dataset)

        for attr in ('targets', 'labels', 'y'):
            if hasattr(base, attr):
                labels = getattr(base, attr)
                if indices is not None:
                    return [int(labels[i]) for i in indices]
                return [int(x) for x in labels]

        if hasattr(base, 'samples'):
            samples = getattr(base, 'samples')
            if indices is not None:
                return [int(samples[i][1]) for i in indices]
            return [int(x[1]) for x in samples]

        raise ValueError('Cannot infer labels for balanced embedding batches.')

    def __iter__(self):
        """
           Генерирует батчи на протяжении одной эпохи.

           На каждой итерации:
               1. Случайно выбирает ``classes_per_batch`` классов без замены
               2. Для каждого класса сэмплирует ``samples_per_class`` индексов
                  (с заменой, если примеров недостаточно)
               3. Перемешивает итоговый батч и возвращает его через yield

           Yields:
               list[int]: индексы примеров для одного батча.
        """
        for _ in range(self.batches_per_epoch):
            selected_classes = np.random.choice(self.classes, size=self.classes_per_batch, replace=False)
            batch: list[int] = []
            for cls in selected_classes:
                indices = self.class_to_indices[int(cls)]
                replace = len(indices) < self.samples_per_class
                chosen = np.random.choice(indices, size=self.samples_per_class, replace=replace)
                batch.extend(int(i) for i in chosen)
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


class BaseTrainer:
    """
    Базовая инфраструктура обучения, общая для всех тренеров.

    Инкапсулирует:
        - перемещение модели и criterion на нужный device
        - инициализацию TensorBoard SummaryWriter
        - Automatic Mixed Precision (AMP) через ``torch.amp.GradScaler``
        - построение DataLoader с настройками из конфига
        - инициализацию оптимизаторов (AdamW, Adam, RAdam, NAdam, Adamax, SGD, RMSprop)
        - инициализацию планировщиков LR (cosine, cosine_with_warmup, StepLR, ReduceOnPlateau, ...)
        - backward + grad clipping + optimizer step
        - основной цикл обучения: ``_run_loop``
        - сохранение checkpoint-а по завершении: ``_save_final_inference_checkpoint``
        - Model Soup (усреднение лучших checkpoint-ов): ``apply_model_soup``
        - построение графиков loss после обучения: ``_plot_loss_curves_after_fit``

    Подклассы обязаны реализовать:
        - ``train_epoch(optimizer, loader) → (train_loss, elapsed_seconds)``
        - ``valid_epoch(loader) → (val_loss, metrics_dict | None, elapsed_seconds)``

    Конфигурация (из ConfigReader / train_conf.yaml):
        TRAINER:
            batch_size (int, 64):         размер батча
            num_workers (int, 8):         число воркеров DataLoader
            shuffle (bool, True):         перемешивание train-лоадера
            use_amp (bool, True):         включить Automatic Mixed Precision
            num_epochs (int, 100):        число эпох
            lr (float, 1e-3):             начальный learning rate
            optimizer_type (str, 'AdamW'): тип оптимизатора
            weight_decay (float, 0.05):   L2-регуляризация для decay-параметров
            scheduler_type (str, 'cosine'): тип планировщика LR
            warmup_epochs (int, 0):       эпохи прогрева (для cosine_with_warmup)
            train_mode (str):             'train' | 'resume' | 'finetune'
            load_checkpoint (str):        путь к .pth для resume/finetune

    Аргументы конструктора:
        model (nn.Module):     модель (будет перемещена на device).
        criterion (nn.Module): функция потерь (будет перемещена на device;
                                её параметры включаются в оптимизатор
                                для обучаемых лоссов — ArcFace, ProxyNCA и др.).
        train_dataset:         тренировочный датасет.
        val_dataset:           валидационный датасет.

    Атрибуты:
        device (torch.device):              cuda если доступен, иначе cpu.
        use_amp (bool):                     флаг AMP.
        scaler (GradScaler | None):         скейлер для AMP.
        runs_dir (str):                     путь к директории TensorBoard-логов.
        writers (list[SummaryWriter]):      TensorBoard-писатели.
        train_loss_history (list[float]):   train loss по эпохам.
        val_loss_history (list[float]):     val loss по эпохам.
        loss_epoch_history (list[int]):     номера эпох для графиков.
    """

    def __init__(self, model, criterion, train_dataset, val_dataset,):
        self.cfg = ConfigReader()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_loss_history: list = []
        self.train_loss_history: list = []
        self.loss_epoch_history: list[int] = []
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self._project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.runs_dir = os.path.join(self._project_root, 'runs', run_name)
        self.writers = [SummaryWriter(log_dir=self.runs_dir)]
        logger.info(f'TensorBoard logs → {self.runs_dir}')
        self.use_amp = bool(self.cfg.get('TRAINER', 'use_amp', True))
        self.scaler = torch.amp.GradScaler(device=self.device.type) if self.use_amp else None

        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info(f'{self.__class__.__name__} initialized | device={self.device} | AMP={self.use_amp} | '
                    f'model={model.__class__.__name__}')

    @staticmethod
    def _unwrap_base_dataset(ds):
        """
        Разворачивает датасет до базового (non-Subset) уровня.

        Нужно для доступа к атрибутам (classes, class_to_idx) у датасетов,
        обёрнутых в ``torch.utils.data.Subset`` при train/val-разбиении.

        Аргументы:
            ds: датасет или Subset.

        Возвращает:
            Базовый датасет без обёрток Subset.
        """
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        return ds

    def _infer_num_classes(self) -> int | None:
        """
        Определяет число классов из датасета или конфига.

        Порядок поиска:
            1. ``dataset.classes``     — list имён классов (ImageFolder и др.)
            2. ``dataset.class_to_idx``— dict {name: idx}
            3. ``DATASET.num_classes`` — явное значение в конфиге

        Возвращает:
            int если удалось определить, None иначе.

        Используется:
            - ``TrainerEmbeddings.build_dataloaders`` для расчёта classes_per_batch
            - ``TrainerEmbeddings._plot_embedding_class_centroids_after_fit``
        """
        for ds in (self.train_dataset, self.val_dataset):
            base = self._unwrap_base_dataset(ds)
            if getattr(base, 'classes', None):
                return len(base.classes)
            m = getattr(base, 'class_to_idx', None)
            if m:
                return len(m)
        cfg_num = self.cfg.get('DATASET', 'num_classes', None)
        if cfg_num is not None and cfg_num != 'null':
            try:
                return int(cfg_num)
            except (TypeError, ValueError):
                pass
        return None

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Строит стандартные DataLoader-ы из конфига.

        Настройки из ``TRAINER``:
            batch_size (int, 64):    размер батча
            num_workers (int, 8):   число параллельных воркеров
            shuffle (bool, True):   перемешивание тренировочного лоадера

        ``pin_memory`` включается автоматически при ``device=cuda``.
        ``persistent_workers`` включается при ``num_workers > 0``, что
        исключает пересоздание воркеров между эпохами.

        Возвращает:
            tuple[DataLoader, DataLoader]: (train_loader, val_loader)
        """
        batch_size = int(self.cfg.get('TRAINER', 'batch_size', 64))
        num_workers = int(self.cfg.get('TRAINER', 'num_workers', 8))
        shuffle_train = bool(self.cfg.get('TRAINER', 'shuffle', True))

        train_ds, val_ds = self.train_dataset, self.val_dataset

        pin = self.device.type == 'cuda'
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent_workers,
        )

        logger.info(f'DataLoaders | train={len(train_ds)} | val={len(val_ds)} | batch_size={batch_size}')
        return train_loader, val_loader

    def init_optimizer(self, model, lr):
        """
        Создаёт оптимизатор с разделением параметров на decay / no-decay группы.

        Decay-группа (weight_decay применяется):
            все параметры, у которых ``requires_grad=True``
            И имя не содержит 'bias' или 'norm'.

        No-decay группа (weight_decay=0.0):
            bias-параметры и norm-слои (BatchNorm, LayerNorm, GroupNorm).

        Параметры criterion также включаются в оптимизатор — нужно для
        обучаемых лоссов (ArcFace, ProxyNCA, ProxyAnchor и др.).

        Поддерживаемые оптимизаторы (TRAINER.optimizer_type):
            'adamw'   — AdamW (betas, eps, amsgrad)
            'adam'    — Adam  (betas, eps, amsgrad)
            'sgd'     — SGD   (momentum, nesterov)
            'rmsprop' — RMSprop (momentum, eps)
            'radam'   — RAdam (betas, eps)
            'nadam'   — NAdam (betas, eps)
            'adamax'  — Adamax (betas, eps)

        Аргументы:
            model: модуль с параметрами (обычно self.model).
            lr (float): начальный learning rate.

        Возвращает:
            torch.optim.Optimizer: инициализированный оптимизатор.

        Исключения:
            ValueError: нет обучаемых параметров или неизвестный optimizer_type.
        """
        optimizer_type = str(self.cfg.get('TRAINER', 'optimizer_type', 'AdamW')).lower()
        weight_decay = float(self.cfg.get('TRAINER', 'weight_decay', 0.05))

        named_params = list(model.named_parameters())
        named_params.extend((f'criterion.{n}', p) for n, p in self.criterion.named_parameters())

        decay_params = [p for n, p in named_params
                        if p.requires_grad and not any(nd in n for nd in ['bias', 'norm'])]
        no_decay_params = [p for n, p in named_params
                           if p.requires_grad and any(nd in n for nd in ['bias', 'norm'])]
        if not decay_params and not no_decay_params:
            raise ValueError('No trainable parameters found in model or criterion.')

        betas_raw = self.cfg.get('OPTIMIZER', 'betas', [0.9, 0.999])
        betas = tuple(betas_raw) if isinstance(betas_raw, (list, tuple)) else (0.9, 0.999)
        eps = float(self.cfg.get('OPTIMIZER', 'eps', 1e-8))
        amsgrad = bool(self.cfg.get('OPTIMIZER', 'amsgrad', False))
        momentum = float(self.cfg.get('OPTIMIZER', 'momentum', 0.9))
        nesterov = bool(self.cfg.get('OPTIMIZER', 'nesterov', True))

        if optimizer_type == 'adamw':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
        elif optimizer_type == 'adam':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
        elif optimizer_type == 'sgd':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov)
        elif optimizer_type == 'rmsprop':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.RMSprop(param_groups, lr=lr, momentum=momentum, eps=eps)
        elif optimizer_type == 'radam':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.RAdam(param_groups, lr=lr, betas=betas, eps=eps)
        elif optimizer_type == 'nadam':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.NAdam(param_groups, lr=lr, betas=betas, eps=eps)
        elif optimizer_type == 'adamax':
            param_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            optimizer = torch.optim.Adamax(param_groups, lr=lr, betas=betas, eps=eps)
        else:
            raise ValueError(f'Unknown optimizer_type: {optimizer_type}. '
                             f'Supported: AdamW, Adam, RAdam, NAdam, Adamax, SGD, RMSprop')

        logger.info(f'Optimizer: {optimizer_type.upper()} (lr={lr}, weight_decay={weight_decay})')
        return optimizer

    @staticmethod
    def get_rate(optimizer) -> float:
        """Возвращает текущий LR первой param-группы оптимизатора."""
        return optimizer.param_groups[0]['lr']

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        """
        Возвращает все обучаемые параметры модели и criterion.

        Используется в ``_backward_step`` для grad clipping — гарантирует,
        что clipping применяется ко всем параметрам, включая веса ArcFace/ProxyNCA.
        """
        return [p for p in list(self.model.parameters()) + list(self.criterion.parameters()) if p.requires_grad]

    def _backward_step(self, loss: torch.Tensor, optimizer) -> None:
        """
        Выполняет backward pass с grad clipping и шагом оптимизатора.

        При AMP (use_amp=True):
            1. ``scaler.scale(loss).backward()``
            2. ``scaler.unscale_(optimizer)``    — масштаб удалён перед clipping-ом
            3. ``clip_grad_norm_(..., max_norm=1.0)``
            4. ``scaler.step(optimizer)``
            5. ``scaler.update()``

        Без AMP:
            1. ``loss.backward()``
            2. ``clip_grad_norm_(..., max_norm=1.0)``
            3. ``optimizer.step()``

        Grad clipping (max_norm=1.0) предотвращает взрывной рост градиентов
        при обучении трансформеров и глубоких backbone-ов.

        .. note::
            ``optimizer.zero_grad()`` должен вызываться **до** этого метода,
            что сделано в ``train_epoch`` каждого подкласса.

        Аргументы:
            loss (torch.Tensor): скалярный лосс текущего батча.
            optimizer:           инициализированный оптимизатор.
        """
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), max_norm=1.0)
            optimizer.step()

    def init_scheduler(self, optimizer, lr, num_epochs, scheduler_type='cosine'):
        """
        Создаёт планировщик learning rate.

        Поддерживаемые типы (TRAINER.scheduler_type):
            'cosine_with_warmup':
                Линейный прогрев на ``warmup_epochs`` эпох, затем cosine annealing.
                Реализован через ``LambdaLR`` для гибкой настройки.
                Параметры: warmup_epochs, warmup_start_factor (0.01), eta_min.

            'cosine':
                ``CosineAnnealingLR(T_max=num_epochs, eta_min)``.

            'steplr':
                ``StepLR(step_size, gamma)``.

            'multisteplr':
                ``MultiStepLR(milestones=[30,60,90], gamma)``.

            'reduceonplateau':
                ``ReduceLROnPlateau(mode, factor, patience, min_lr)``.
                Требует передачи val_loss в ``scheduler.step(val_loss)``
                — обрабатывается автоматически в ``_run_loop``.

            'exponential':
                ``ExponentialLR(gamma=exp_gamma)``.

            'none' / '':
                Без планировщика, LR остаётся постоянным.

        Аргументы:
            optimizer:            инициализированный оптимизатор.
            lr (float):           начальный LR (нужен для cosine_with_warmup).
            num_epochs (int):     полное число эпох обучения.
            scheduler_type (str): тип планировщика из конфига.

        Возвращает:
            torch.optim.lr_scheduler._LRScheduler | None
        """
        warmup_epochs = int(self.cfg.get('TRAINER', 'warmup_epochs', 0))

        step_size = int(self.cfg.get('SCHEDULER', 'step_size', 10))
        gamma = float(self.cfg.get('SCHEDULER', 'gamma', 0.1))
        milestones_raw = self.cfg.get('SCHEDULER', 'milestones', [30, 60, 90])
        milestones = list(milestones_raw) if milestones_raw else [30, 60, 90]

        plateau_mode = str(self.cfg.get('SCHEDULER', 'plateau_mode', 'min'))
        plateau_factor = float(self.cfg.get('SCHEDULER', 'plateau_factor', 0.5))
        plateau_patience = int(self.cfg.get('SCHEDULER', 'plateau_patience', 7))
        plateau_min_lr = float(self.cfg.get('SCHEDULER', 'plateau_min_lr', 1e-7))

        exp_gamma = float(self.cfg.get('SCHEDULER', 'exp_gamma', 0.95))
        eta_min = float(self.cfg.get('SCHEDULER', 'eta_min', 0))
        warmup_start_factor = float(self.cfg.get('SCHEDULER', 'warmup_start_factor', 0.01))

        s = scheduler_type.lower()

        if s == 'cosine_with_warmup':
            warmup_epochs = max(0, min(warmup_epochs, max(0, num_epochs - 1)))
            cosine_epochs = max(1, num_epochs - warmup_epochs)
            min_factor = eta_min / lr if lr > 0 else 0.0
            start_factor = max(0.0, min(1.0, warmup_start_factor))

            def lr_lambda(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return start_factor + (1.0 - start_factor) * (epoch / float(warmup_epochs))
                progress = min(1.0, (epoch - warmup_epochs) / float(cosine_epochs))
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine_factor

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif s == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, num_epochs), eta_min=eta_min
            )
        elif s == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif s == 'multisteplr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
        elif s == 'reduceonplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=plateau_mode, factor=plateau_factor,
                patience=plateau_patience, min_lr=plateau_min_lr
            )
        elif s == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=exp_gamma
            )
        elif s == 'none' or s == '':
            scheduler = None
        else:
            logger.warning(f'Unknown scheduler_type: {scheduler_type}. Using None.')
            scheduler = None

        logger.info(f'Scheduler: {scheduler_type if scheduler else "None"}')
        return scheduler

    def train_epoch(self, optimizer, loader) -> tuple[float, float]:
        """
        Одна тренировочная эпоха. Реализуется в подклассе.

        Возвращает:
            tuple[float, float]: (средний train_loss за эпоху, время в секундах)
        """
        raise NotImplementedError

    def valid_epoch(self, loader) -> tuple:
        """
        Одна валидационная эпоха. Реализуется в подклассе.

        Возвращает:
            tuple: (val_loss, metrics_dict | None, время в секундах)
        """
        raise NotImplementedError

    def _format_epoch_output(self, epoch: int, num_epochs: int, train_loss: float, val_loss: float,
                             lr: float, metrics: dict | None, train_time: float, val_time: float) -> str:
        """
        Форматирует красивый вывод строк эпохи в box-стиле (Unicode).

        Пример вывода:
            ╔══════════════════════════════════════════════════════════════════╗
            ║  Epoch    5 / 100  │  train_loss: 0.4231  │  val_loss: 0.4891  ║
            ╠══════════════════════════════════════════════════════════════════╣
            ║  ⏱ Train: 12.3s  │  Val: 3.1s                                  ║
            ╠══════════════════════════════════════════════════════════════════╣
            ║  accuracy: 0.8721  │  f1_macro: 0.8643                          ║
            ╚══════════════════════════════════════════════════════════════════╝

        Метрики выводятся по 3 штуки в строку. confusion_matrix пропускается
        (выводится отдельно в TrainerClassification._format_epoch_output).

        Аргументы:
            epoch (int):          номер текущей эпохи.
            num_epochs (int):     всего эпох.
            train_loss (float):   средний лосс на трейне.
            val_loss (float):     средний лосс на вале.
            lr (float):           текущий learning rate.
            metrics (dict | None): словарь метрик от valid_epoch.
            train_time (float):   время тренировочной эпохи в секундах.
            val_time (float):     время валидационной эпохи в секундах.

        Возвращает:
            str: многострочный форматированный вывод.
        """
        width = 92

        def box_line(s: str, w: int = width - 4) -> str:
            content = ('  ' + s)[:w].ljust(w)
            return '║' + content + '║'

        lines = [
            '',
            '╔' + '═' * (width - 2) + '╗',
            box_line(
                f'Epoch {epoch:>4} / {num_epochs}  │  train_loss: {train_loss:.4f}  │  val_loss: {val_loss:.4f}  │  lr: {lr:.2e}'),
            '╠' + '═' * (width - 2) + '╣',
            box_line(f'⏱ Train: {train_time:.1f}s  │  Val: {val_time:.1f}s'),
        ]

        if metrics:
            lines.append('╠' + '═' * (width - 2) + '╣')
            metric_parts = []
            for k, v in metrics.items():
                try:
                    metric_parts.append(f'{k}: {float(v):.4f}')
                except (TypeError, ValueError):
                    continue
            if metric_parts:
                sep = '  │  '
                full = sep.join(metric_parts)
                max_content = width - 4
                if len(full) <= max_content:
                    lines.append(box_line(full))
                else:
                    for i in range(0, len(metric_parts), 3):
                        chunk = sep.join(metric_parts[i:i + 3])
                        lines.append(box_line(chunk))

        lines.append('╚' + '═' * (width - 2) + '╝')
        return '\n'.join(lines)

    def load_checkpoint(self, mode: str = 'train'):
        """
        Загружает checkpoint из .pth файла.

        Восстанавливает:
            - ``model_state_dict``     — веса модели
            - ``criterion_state_dict`` — веса criterion (если есть, для ArcFace и др.)

        После загрузки устанавливает режим модели:
            mode='train' → ``model.train()``
            mode='eval'  → ``model.eval()``

        Аргументы:
            mode (str): 'train' для resume/finetune, 'eval' для инференса.

        Возвращает:
            tuple: (optimizer_state_dict, epoch: int)
                optimizer_state_dict используется при resume для продолжения
                с того же состояния оптимизатора.

        Исключения:
            ValueError: если путь не заканчивается на .pth.
        """
        checkpoint_path = self.cfg.get('TRAINER', 'load_checkpoint', 'weights/best.pth')

        if not checkpoint_path.endswith('.pth'):
            raise ValueError('Checkpoint must be a .pth file')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'criterion_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.model.train() if mode == 'train' else self.model.eval()
        logger.info(f'Checkpoint loaded: {checkpoint_path} '
                    f'[epoch={checkpoint["epoch"]}, mode={mode}]')
        return checkpoint['optimizer_state_dict'], checkpoint['epoch']

    def _save_final_inference_checkpoint(self, optimizer, epoch_last: int) -> None:
        """
        Сохраняет финальный checkpoint после окончания обучения.

        Checkpoint содержит:
            - model_state_dict       — текущие веса в памяти
            - criterion_state_dict   — веса criterion
            - optimizer_state_dict   — состояние оптимизатора
            - epoch                  — последняя завершённая эпоха
            - note                   — текстовое примечание

        .. note::
            Если был применён Model Soup (``apply_model_soup``), веса в памяти
            уже усреднены. Этот файл отражает состояние ПОСЛЕ soup.
            Файл ``best_checkpoint.pth`` (сохраняется EarlyStopping) содержит
            лучшие веса ДО soup.

        Путь сохранения:
            ``{project_root}/weights/{ModelClassName}/final_inference.pth``
        """
        model_name = self.model.__class__.__name__
        save_dir = os.path.join(self._project_root, 'weights', model_name)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'final_inference.pth')

        was_training = self.model.training
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': max(0, int(epoch_last)),
            'note': 'After training (weights in memory match this file; best_checkpoint.pth may differ if soup was applied).',
        }, path)
        if was_training:
            self.model.train()
        logger.info(f'Final inference checkpoint saved → {path}')

    def apply_model_soup(self, early_stopping: EarlyStopping):
        """
        Применяет Model Soup: усредняет веса лучших checkpoint-ов.

        Model Soup (Wortsman et al., 2022) — усреднение весов нескольких
        checkpoint-ов с разных эпох в пространстве весов.
        Как правило, улучшает обобщающую способность без дополнительного обучения.

        Поддерживаемые стратегии (EARLY_STOPPING.soup_strategy):
            'avg': простое арифметическое усреднение весов из
                   ``early_stopping.best_nets`` (список state_dict-ов).

        После применения soup текущие веса модели заменяются усреднёнными.

        Аргументы:
            early_stopping (EarlyStopping): объект, хранящий список
                лучших state_dict-ов (``best_nets``).

        Исключения:
            ValueError: если soup_strategy не равна 'avg'.
        """
        if self.cfg.get('EARLY_STOPPING', 'soup_strategy', 'avg') == 'avg':
            averaged = early_stopping.avg_net()
            if averaged is not None:
                self.model.load_state_dict(averaged)
                logger.info(
                    f'Model Soup applied from {len(early_stopping.best_nets)} checkpoints'
                )
        else:
            raise ValueError('Early stopping strategy must be "avg" another strategy are not implemented yet"')

    def _plot_loss_curves_after_fit(self) -> None:
        """
        Строит и сохраняет график train/val loss после обучения.

        Использует matplotlib в backend='Agg' (без GUI).
        Сохраняет в ``{runs_dir}/loss_curves.png``.
        Пропускает, если история пустая или matplotlib не установлен.
        """
        if not self.train_loss_history or not self.val_loss_history:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning('matplotlib not installed — skip loss curve plot')
            return

        epochs = self.loss_epoch_history or list(range(len(self.train_loss_history)))
        path = os.path.join(self.runs_dir, 'loss_curves.png')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_loss_history, marker='o', linewidth=1.8, label='train_loss')
        plt.plot(epochs, self.val_loss_history, marker='o', linewidth=1.8, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model.__class__.__name__} loss curves')
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()
        logger.info(f'Loss curves saved → {path}')

    def _run_loop(self, optimizer, scheduler, train_loader, val_loader, start_epoch: int, num_epochs: int, early_stopping: EarlyStopping):
        """
        Основной цикл обучения: эпохи, логирование, scheduler, early stopping.

        Алгоритм каждой эпохи:
            1. ``train_epoch`` → train_loss
            2. ``valid_epoch`` → val_loss, metrics_dict
            3. Запись в TensorBoard: Loss/train, Loss/val, LR, все метрики
            4. Форматированный вывод в лог
            5. Шаг scheduler (для ReduceOnPlateau — передаётся val_loss)
            6. Определение monitor-значения для EarlyStopping:
               - monitor='val_loss'          → val_loss (mode='min')
               - monitor=имя метрики         → значение из metrics_dict
               - метрика недоступна          → fallback на val_loss
            7. ``early_stopping.step(...)``  → сохраняет checkpoint если улучшение
            8. Прерывание при ``early_stopping.stop=True``

        После цикла (в блоке finally):
            - Закрываются все TensorBoard-писатели
            - Применяется Model Soup (``apply_model_soup``)
            - Сохраняется финальный checkpoint

        Аргументы:
            optimizer:               инициализированный оптимизатор.
            scheduler:               планировщик LR или None.
            train_loader (DataLoader): тренировочный лоадер.
            val_loader (DataLoader):   валидационный лоадер.
            start_epoch (int):         начальная эпоха (0 или продолжение с checkpoint).
            num_epochs (int):          конечная эпоха (не число эпох, а индекс).
            early_stopping (EarlyStopping): объект для мониторинга и сохранения.
        """
        epoch_last_completed = start_epoch - 1
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_last_completed = epoch
                train_loss, train_time = self.train_epoch(optimizer, train_loader)
                val_loss, metrics_dict, val_time = self.valid_epoch(val_loader)
                current_lr = self.get_rate(optimizer)

                self.train_loss_history.append(train_loss)
                self.val_loss_history.append(val_loss)
                self.loss_epoch_history.append(epoch)
                for w in self.writers:
                    w.add_scalar('Loss/train', train_loss, epoch)
                    w.add_scalar('Loss/val', val_loss, epoch)
                    w.add_scalar('LR', current_lr, epoch)

                if metrics_dict:
                    for k, v in metrics_dict.items():
                        if k != 'confusion_matrix':
                            try:
                                for w in self.writers:
                                    w.add_scalar(f'Metrics/{k}', float(v), epoch)
                            except (TypeError, ValueError):
                                pass
                    for w in self.writers:
                        w.flush()

                output = self._format_epoch_output(
                    epoch, num_epochs, train_loss, val_loss, current_lr,
                    metrics_dict, train_time, val_time
                )
                logger.info(output)

                if scheduler:
                    if isinstance(scheduler,
                                  torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                monitor_name = early_stopping.monitor

                if monitor_name == 'val_loss':
                    monitor_value = val_loss
                    comparison_mode = 'min'
                    metric_log_name = 'val_loss'
                else:
                    comparison_mode = str(early_stopping.mode).lower()
                    monitor_value = None
                    metric_log_name = monitor_name
                    if metrics_dict:
                        monitor_value = metrics_dict.get(monitor_name)
                    if monitor_value is None:
                        logger.warning(
                            f'Early stopping monitor "{monitor_name}" is unavailable; falling back to val_loss (mode=min).'
                        )
                        monitor_value = val_loss
                        comparison_mode = 'min'
                        metric_log_name = 'val_loss'
                    elif isinstance(monitor_value, torch.Tensor):
                        monitor_value = float(monitor_value.detach().float().mean().item())
                    else:
                        monitor_value = float(monitor_value)

                early_stopping.step(
                    self.model, self.criterion, optimizer, epoch, monitor_value, val_loss=val_loss,
                    comparison_mode=comparison_mode,
                    metric_log_name=metric_log_name,
                )
                if early_stopping.stop:
                    break
        finally:
            for w in self.writers:
                w.close()
            logger.info('TensorBoard writer closed')

        self.apply_model_soup(early_stopping)
        self._save_final_inference_checkpoint(optimizer, epoch_last_completed)

        if self.val_loss_history:
            best_val = float(min(self.val_loss_history))
            best_loss_idx = int(np.argmin(self.val_loss_history))
            best_loss_epoch = start_epoch + best_loss_idx
            label = getattr(early_stopping, 'best_metric_log_name', early_stopping.monitor)
            best_m = float(getattr(early_stopping, 'best_score', float('nan')))
            best_ep = int(getattr(early_stopping, 'best_epoch', -1))
            w = 50
            line = f'  Best {label}: {best_m:.4f} at epoch {best_ep}  '
            val_line = f'  Lowest val_loss: {best_val:.4f} at epoch {best_loss_epoch}  '
            box = '\n'.join([
                '+' + '-' * w + '+',
                '|' + ' Training completed '.center(w) + '|',
                '+' + '-' * w + '+',
                '|' + line.ljust(w) + '|',
                '|' + val_line.ljust(w) + '|',
                '+' + '-' * w + '+',
            ])
            print('\n' + box + '\n')

    def fit(self):
        """
        Точка входа для запуска полного цикла обучения.

        Режимы обучения (TRAINER.train_mode):
            'train':    обучение с нуля
            'resume':   продолжение с checkpoint (восстанавливает эпоху и состояние оптимизатора)
            'finetune': дообучение с предобученных весов (оптимизатор инициализируется заново)

        Алгоритм:
            1. Читает конфиг (num_epochs, lr, scheduler_type, optimizer_type)
            2. Печатает информационный box «Training started»
            3. Инициализирует EarlyStopping, optimizer, scheduler, DataLoader-ы
            4. При resume/finetune — загружает checkpoint
            5. Запускает ``_run_loop``
            6. Строит графики loss (``_plot_loss_curves_after_fit``)

        Исключения:
            ValueError: если train_mode не задан в конфиге.
        """
        train_mode = self.cfg.get('TRAINER', 'train_mode', None)
        num_epochs = int(self.cfg.get('TRAINER', 'num_epochs', 100))
        scheduler_type = self.cfg.get('TRAINER', 'scheduler_type', 'cosine')
        lr = float(self.cfg.get('TRAINER', 'lr', 1e-3))

        if train_mode is None:
            raise ValueError('train_mode must be set in train_conf.yaml')

        optimizer_type = self.cfg.get('TRAINER', 'optimizer_type', 'AdamW')
        content = f'  mode: {train_mode}  |  epochs: {num_epochs}  |  optimizer: {optimizer_type}  |  scheduler: {scheduler_type}  '
        w = max(len(content), 58)
        box = '\n'.join([
            '+' + '-' * w + '+',
            '|' + ' Training started '.center(w) + '|',
            '+' + '-' * w + '+',
            '|' + content.ljust(w) + '|',
            '+' + '-' * w + '+',
        ])
        print('\n' + box + '\n')

        model_name = self.model.__class__.__name__
        early_stopping = EarlyStopping(model_name=model_name)
        optimizer = self.init_optimizer(self.model, lr)
        scheduler = self.init_scheduler(optimizer, lr, num_epochs, scheduler_type)
        train_loader, val_loader = self.build_dataloaders()
        self._last_val_loader = val_loader

        start_epoch = 0
        if train_mode == 'resume':
            opt_state, start_epoch = self.load_checkpoint(mode='train')
            optimizer.load_state_dict(opt_state)
            logger.info(f'Resuming from epoch {start_epoch}')
        elif train_mode == 'finetune':
            self.load_checkpoint(mode='train')
            logger.info('Fine-tuning from pretrained checkpoint')

        self._run_loop(optimizer, scheduler, train_loader, val_loader,
                       start_epoch, num_epochs, early_stopping)
        self._plot_loss_curves_after_fit()


class TrainerClassification(BaseTrainer):
    """
        Тренер для задачи классификации.

        Наследует всю инфраструктуру BaseTrainer и реализует:
            - ``train_epoch``: forward → CrossEntropy (или любой criterion) → backward
            - ``valid_epoch``: forward → loss + sklearn-метрики + confusion matrix
            - ``_format_epoch_output``: вывод confusion matrix после метрик

        Поддерживает 5D-тензоры ``(B, T, C, H, W)`` — батч видеоклипов / аугментаций.
        В этом случае все T кадров прогоняются независимо, логиты усредняются по T.

        Конфигурация (METRICS):
            enabled (bool, False):      включить вычисление метрик
            list (list[str]):           список имён метрик из METRIC_FUNCS
            class_names (list[str]):    имена классов для confusion matrix

        Пример:
            trainer = TrainerClassification(model, nn.CrossEntropyLoss(), train_ds, val_ds)
            trainer.fit()
        """

    def _forward_logits(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Прогоняет батч через модель и возвращает логиты.

        Обрабатывает 5D-тензоры (видео/мульти-кроп аугментации):
            (B, T, C, H, W) → reshape (B*T, C, H, W) → model → усреднение по T → (B, num_classes)

        Аргументы:
            x (torch.Tensor): батч изображений (B, C, H, W) или (B, T, C, H, W).
            y (torch.Tensor): метки классов (B,) — нужны для восстановления B из B*T.

        Возвращает:
            torch.Tensor: логиты (B, num_classes).
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x_ = x.view(b * t, c, h, w)
        else:
            x_ = x

        logits = self.model(x_)
        if logits.size(0) != y.size(0):
            logits = logits.view(y.size(0), -1, logits.size(-1)).mean(dim=1)
        return logits

    def train_epoch(self, optimizer, loader) -> tuple[float, float]:
        """
        Одна тренировочная эпоха классификации.

        Шаги:
            1. ``model.train()``
            2. Для каждого батча: forward → criterion(logits, y) → backward_step
            3. Возврат среднего лосса и времени

        Возвращает:
            tuple[float, float]: (avg_train_loss, elapsed_seconds)
        """
        SINCE = time.time()

        self.model.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc='Training classification', leave=False, file=sys.stdout,
                         dynamic_ncols=False, ncols=100, miniters=1, smoothing=0):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            def _forward():
                logits = self._forward_logits(x, y)
                return self.criterion(logits, y)

            if self.use_amp:
                with torch.autocast(device_type=self.device.type):
                    loss = _forward()
            else:
                loss = _forward()

            self._backward_step(loss, optimizer)
            total_loss += loss.item()

        time_elapsed = time.time() - SINCE

        logger.info('Classification training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        ))
        return total_loss / len(loader), time_elapsed

    def _get_metrics_config(self):
        """
        Читает настройки классификационных метрик из конфига.

        Возвращает:
            tuple: (enabled: bool, metrics_list: list[str], class_names: list[str])
        """
        enabled = bool(self.cfg.get('METRICS', 'enabled', False))
        metrics_list = self.cfg.get('METRICS', 'list', [])
        class_names = self.cfg.get('METRICS', 'class_names', [])
        if metrics_list is not None and hasattr(metrics_list, '__iter__') and not isinstance(metrics_list, str):
            metrics_list = OmegaConf.to_container(metrics_list, resolve=True) if metrics_list else []
        else:
            metrics_list = []
        if class_names is not None and hasattr(class_names, '__iter__') and not isinstance(class_names, str):
            class_names = OmegaConf.to_container(class_names, resolve=True) if class_names else []
        else:
            class_names = []
        return enabled, list(metrics_list or []), list(class_names or [])

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics_list: list, class_names: list) -> dict:
        """
        Вычисляет метрики классификации через sklearn.

        Поддерживаемые метрики (из METRIC_FUNCS + confusion_matrix):
            accuracy, precision_{macro,micro,weighted},
            recall_{macro,micro,weighted}, f1_{macro,micro,weighted},
            confusion_matrix

        Аргументы:
            y_true (np.ndarray):   истинные метки (N,).
            y_pred (np.ndarray):   предсказанные метки (N,).
            metrics_list (list):   список имён метрик.
            class_names (list):    имена классов (используются при выводе).

        Возвращает:
            dict: {metric_name: value}. confusion_matrix → np.ndarray.
        """
        result = {}
        for name in metrics_list:
            if name == 'confusion_matrix':
                cm = confusion_matrix(y_true, y_pred)
                result[name] = cm
            elif name in METRIC_FUNCS:
                result[name] = float(METRIC_FUNCS[name](y_true, y_pred))
        return result

    def valid_epoch(self, loader) -> tuple:
        """
        Одна валидационная эпоха классификации.

        Шаги:
            1. ``model.eval()`` + ``torch.no_grad()``
            2. Для каждого батча: forward → loss + argmax → накопление предсказаний
            3. После цикла: вычисление метрик по накопленным y_true / y_pred

        Возвращает:
            tuple: (avg_val_loss: float, metrics_dict: dict | None, elapsed: float)
        """
        SINCE = time.time()
        metrics_enabled, metrics_list, _ = self._get_metrics_config()
        all_preds, all_labels = [], []

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(loader, desc='Validation classification', leave=False, file=sys.stdout,
                             dynamic_ncols=False, ncols=100, miniters=1, smoothing=0):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                def _val_forward():
                    logits = self._forward_logits(x, y)
                    return self.criterion(logits, y), logits

                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        loss, logits = _val_forward()
                else:
                    loss, logits = _val_forward()

                total_loss += loss.item()

                if metrics_enabled and metrics_list:
                    pred_cls = logits.argmax(dim=1).cpu().numpy()
                    all_preds.append(pred_cls)
                    all_labels.append(y.cpu().numpy())

        time_elapsed = time.time() - SINCE
        val_loss = total_loss / len(loader)

        metrics_dict = None
        if metrics_enabled and metrics_list and all_preds:
            y_true = np.concatenate(all_labels)
            y_pred = np.concatenate(all_preds)
            _, _, class_names = self._get_metrics_config()
            metrics_dict = self._compute_metrics(y_true, y_pred, metrics_list, class_names)

        logger.info('Classification validation complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60
        ))
        return val_loss, metrics_dict, time_elapsed

    def _format_epoch_output(self, epoch: int, num_epochs: int, train_loss: float, val_loss: float,
                             lr: float, metrics: dict | None, train_time: float, val_time: float) -> str:
        """
        Расширяет вывод BaseTrainer confusion matrix-ом.

        Если в метриках есть 'confusion_matrix', добавляет его
        в виде текстовой таблицы с именами классов после основного box.
        """
        output = super()._format_epoch_output(
            epoch, num_epochs, train_loss, val_loss, lr, metrics, train_time, val_time
        )
        if not metrics or 'confusion_matrix' not in metrics:
            return output

        _, _, class_names = self._get_metrics_config()
        cm = metrics['confusion_matrix']
        names = class_names[:cm.shape[0]] if class_names else [str(i) for i in range(cm.shape[0])]

        lines = ['Confusion Matrix:']
        header = '          ' + '  '.join(f'{n:>8}' for n in names)
        lines.append(header)
        for i, row in enumerate(cm):
            row_str = f'{names[i]:>8}: ' + '  '.join(f'{int(x):>8}' for x in row)
            lines.append(row_str)
        return output + '\n' + '\n'.join(lines)


class TrainerEmbeddings(BaseTrainer):
    """
    Тренер для metric learning: обучение эмбеддингов.

    Использует ``criterion(embeddings, labels)`` — любой metric learning лосс:
    SupConLoss, TripletLoss, ArcFace, CosFace, ProxyNCA, ProxyAnchor и др.

    Ключевые особенности:
        - ``BalancedClassBatchSampler`` — балансировка батчей по классам
          для гарантированного наличия позитивных пар
        - Оставляет backbone в ``eval()`` при заморозке во время ``train_epoch``
        - Метрики на валидации вычисляются без sklearn — только на эмбеддингах:
          positive_similarity, negative_similarity, nn_accuracy, Recall@K
        - После обучения строит 2D-визуализацию эмбеддингов и центроидов классов

    Конфигурация:
        EMBEDDING_LOSS:
            balanced_batches (bool, True):       использовать BalancedClassBatchSampler
            classes_per_batch (int):             число классов в батче
            samples_per_class (int):             число примеров на класс

        EMBEDDING_METRICS:
            enabled (bool, True):                вычислять embedding-метрики на вале
            plot_centroids_after_fit (bool, True): строить 2D-визуализацию после обучения
            save_centroids_after_fit (bool, True): сохранять .npy центроидов

    Пример:
        trainer = TrainerEmbeddings(model, SupConLoss(), train_ds, val_ds)
        trainer.fit()
        # → {runs_dir}/embeddings/{ModelName}/embedding_clusters.png
    """

    def __init__(self, model, criterion, train_dataset, val_dataset):
        super().__init__(model, criterion, train_dataset, val_dataset)

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Строит DataLoader-ы с опциональной балансировкой классов.

        При ``balanced_batches=True`` (по умолчанию):
            train_loader использует ``BalancedClassBatchSampler`` вместо shuffle.
            Размер батча = classes_per_batch × samples_per_class.

            classes_per_batch по умолчанию:
                min(n_classes, batch_size // 2)

            samples_per_class по умолчанию:
                max(2, batch_size // classes_per_batch)

        При ``balanced_batches=False``:
            делегирует в BaseTrainer.build_dataloaders() — стандартный режим.

        val_loader всегда стандартный (без балансировки).

        Возвращает:
            tuple[DataLoader, DataLoader]: (train_loader, val_loader)
        """
        use_balanced = bool(self.cfg.get('EMBEDDING_LOSS', 'balanced_batches', True))
        if not use_balanced:
            return super().build_dataloaders()

        batch_size = int(self.cfg.get('TRAINER', 'batch_size', 64))
        num_workers = int(self.cfg.get('TRAINER', 'num_workers', 8))
        n_classes = self._infer_num_classes() or 2
        classes_per_batch = int(self.cfg.get('EMBEDDING_LOSS', 'classes_per_batch', min(n_classes, batch_size // 2)))
        classes_per_batch = max(2, min(classes_per_batch, n_classes))
        samples_per_class = int(
            self.cfg.get('EMBEDDING_LOSS', 'samples_per_class', max(2, batch_size // classes_per_batch))
        )

        sampler = BalancedClassBatchSampler(
            self.train_dataset,
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class,
        )

        pin = self.device.type == 'cuda'
        persistent_workers = num_workers > 0
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persistent_workers,
        )

        effective_batch = classes_per_batch * samples_per_class
        logger.info(
            f'Embedding DataLoaders | train={len(self.train_dataset)} | val={len(self.val_dataset)} | '
            f'balanced_batch={classes_per_batch}x{samples_per_class}={effective_batch}'
        )
        return train_loader, val_loader

    def _forward_embeddings(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Прогоняет батч через модель и возвращает эмбеддинги.

        Обрабатывает 5D-тензоры (видео/мульти-кроп):
            (B, T, C, H, W) → reshape (B*T, C, H, W) → model → усреднение по T → (B, D)

        Аргументы:
            x (torch.Tensor): батч изображений (B, C, H, W) или (B, T, C, H, W).
            y (torch.Tensor): метки (B,) — для восстановления B из B*T.

        Возвращает:
            torch.Tensor: матрица эмбеддингов (B, embedding_dim).
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x_ = x.view(b * t, c, h, w)
        else:
            x_ = x

        embeddings = self.model(x_)
        if embeddings.size(0) != y.size(0):
            embeddings = embeddings.view(y.size(0), -1, embeddings.size(-1)).mean(dim=1)
        return embeddings

    def train_epoch(self, optimizer, loader) -> tuple[float, float]:
        """
        Одна тренировочная эпоха metric learning.

        Особенности:
            - После ``model.train()`` backbone переводится в ``eval()`` если заморожен.
              Это предотвращает обновление running_mean/var в BatchNorm и нестабильность
              эмбеддингов при малых батчах.
            - ``criterion(embeddings, labels)`` — метрика-лосс поверх эмбеддингов.

        Backbone остаётся в ``eval()`` если ``any(p.requires_grad for p in backbone.parameters()) == False``.
        Если backbone частично разморожен (finetune) — переводится в ``train()``.

        Возвращает:
            tuple[float, float]: (avg_train_loss, elapsed_seconds)
        """
        SINCE = time.time()

        self.model.train()

        if hasattr(self.model, 'backbone'):
            backbone_trainable = any(p.requires_grad for p in self.model.backbone.parameters())
            self.model.backbone.train(backbone_trainable)

        total_loss = 0.0
        for x, y in tqdm(loader, desc='Training emb_loss', leave=False, file=sys.stdout,
                         dynamic_ncols=False, ncols=100, miniters=1, smoothing=0):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            def _forward():
                embeddings = self._forward_embeddings(x, y)
                return self.criterion(embeddings, y)

            if self.use_amp:
                with torch.autocast(device_type=self.device.type):
                    loss = _forward()
            else:
                loss = _forward()

            self._backward_step(loss, optimizer)
            total_loss += loss.item()

        time_elapsed = time.time() - SINCE

        logger.info('Embedding training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return total_loss / len(loader), time_elapsed

    @staticmethod
    def _compute_embedding_metrics(embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Вычисляет метрики качества эмбеддингов на валидационном множестве.

        Все метрики вычисляются без материализации полной матрицы N×N сходств,
        что позволяет работать с большими датасетами.

        Метрики:
            positive_similarity (float):
                Средний косинусный similarity между эмбеддингами одного класса.
                Вычисляется по случайной выборке пар (до 16384).
                Цель: стремиться к 1.0.

            negative_similarity (float):
                Средний косинусный similarity между эмбеддингами разных классов.
                Цель: стремиться к 0.0 (или отрицательным значениям).

            nn_accuracy (float):
                Nearest-neighbour accuracy: доля примеров, чей ближайший сосед
                (по косинусному similarity, исключая сам пример) принадлежит
                тому же классу.
                Вычисляется через chunked matmul (чанки по 512 строк), что
                исключает хранение матрицы N×N в памяти.
                Цель: стремиться к 1.0.

            recall_at_1, recall_at_5, recall_at_10 (float):
                Recall@K из ``_embedding_retrieval_metrics_at_k``.

        Аргументы:
            embeddings (torch.Tensor): сырые или L2-нормализованные эмбеддинги (N, D).
            labels (torch.Tensor):     метки классов (N,).

        Возвращает:
            dict: {metric_name: float}. Пустой dict если N < 2.
        """
        metrics = {}
        if embeddings.size(0) < 2:
            return metrics

        z = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        lab_long = labels.view(-1).long()

        # Без матрицы N×N и без O(N²) времени по «всем парам»: случайное сублинейное число пар
        n = z.size(0)
        rng_cpu = torch.Generator()
        rng_cpu.manual_seed(42)
        n_pairs = min(16384, max(512, n * 8))
        pos_sum = 0.0
        neg_sum = 0.0
        pos_n = 0
        neg_n = 0
        for _ in range(n_pairs):
            i = int(torch.randint(0, n, (1,), generator=rng_cpu).item())
            j = int(torch.randint(0, n, (1,), generator=rng_cpu).item())
            if i == j:
                continue
            s = float((z[i] * z[j]).sum().item())
            if lab_long[i].item() == lab_long[j].item():
                pos_sum += s
                pos_n += 1
            else:
                neg_sum += s
                neg_n += 1
        if pos_n:
            metrics['positive_similarity'] = pos_sum / float(pos_n)
        if neg_n:
            metrics['negative_similarity'] = neg_sum / float(neg_n)

        # Nearest-neighbor accuracy (exclude self): chunked matmul avoids N×N materialization
        nearest = torch.empty(n, dtype=torch.long, device=z.device)
        chunk_rows = min(512, max(1, n))
        for rs in range(0, n, chunk_rows):
            re = min(n, rs + chunk_rows)
            block = z[rs:re] @ z.t()
            rr = torch.arange(rs, re, device=z.device)
            block[rr - rs, rr] = -float('inf')
            nearest[rs:re] = block.argmax(dim=1)
        metrics['nn_accuracy'] = float((lab_long[nearest] == lab_long).float().mean().item())

        metrics.update(_embedding_retrieval_metrics_at_k(embeddings, labels, ks=(1, 5, 10)))
        return metrics

    def valid_epoch(self, loader) -> tuple:
        """
        Одна валидационная эпоха metric learning.

        Шаги:
            1. ``model.eval()`` + ``torch.no_grad()``
            2. Накапливает эмбеддинги и метки всего val-множества
            3. Вычисляет embedding-метрики через ``_compute_embedding_metrics``

        Возвращает:
            tuple: (avg_val_loss: float, metrics_dict: dict | None, elapsed: float)
        """
        SINCE = time.time()
        metrics_enabled = bool(self.cfg.get('EMBEDDING_METRICS', 'enabled', True))
        all_embeddings, all_labels = [], []

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(loader, desc='Validation emb_loss', leave=False, file=sys.stdout,
                             dynamic_ncols=False, ncols=100, miniters=1, smoothing=0):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                def _val_forward():
                    embeddings = self._forward_embeddings(x, y)
                    return self.criterion(embeddings, y), embeddings

                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        loss, embeddings = _val_forward()
                else:
                    loss, embeddings = _val_forward()

                total_loss += loss.item()
                if metrics_enabled:
                    all_embeddings.append(embeddings.detach().cpu())
                    all_labels.append(y.detach().cpu())

        time_elapsed = time.time() - SINCE
        val_loss = total_loss / len(loader)

        metrics_dict = None
        if metrics_enabled and all_embeddings:
            embeddings = torch.cat(all_embeddings, dim=0)
            labels = torch.cat(all_labels, dim=0)
            metrics_dict = self._compute_embedding_metrics(embeddings, labels)

        logger.info('Embedding validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return val_loss, metrics_dict, time_elapsed

    @torch.no_grad()
    def _embeddings_labels_from_loader(self, loader):
        """
        Прогоняет весь лоадер и собирает эмбеддинги и метки.

        Используется для построения 2D-визуализации после обучения.
        Модель переводится в ``eval()`` перед прогоном.

        Возвращает:
            tuple[torch.Tensor, torch.Tensor]:
                z   — матрица эмбеддингов (N, D) на CPU
                lab — метки классов (N,) long на CPU
        """
        zs, ys = [], []
        self.model.eval()
        for x, y in tqdm(
                loader, desc='Gathering embeddings (viz)', leave=False,
                file=sys.stdout, dynamic_ncols=False, ncols=100, miniters=1, smoothing=0,
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            emb = self._forward_embeddings(x, y).detach().float().cpu()
            zs.append(emb)
            ys.append(y.cpu())
        z = torch.cat(zs, dim=0)
        lab = torch.cat(ys, dim=0).view(-1).long()
        return z, lab

    def _embedding_class_centroids(self, embeddings_norm: torch.Tensor, labels_long: torch.Tensor, n_classes: int) -> tuple[np.ndarray, list[str]]:
        """
        Вычисляет L2-нормализованные центроиды классов в пространстве эмбеддингов.

        Алгоритм:
            1. L2-нормализует все эмбеддинги
            2. Для каждого класса усредняет эмбеддинги по маске
            3. L2-нормализует центроид

        Пропускает классы с нулём примеров (отсутствующие в батче).

        Аргументы:
            embeddings_norm (torch.Tensor): эмбеддинги (N, D).
            labels_long (torch.Tensor):     метки (N,) long.
            n_classes (int):                число классов.

        Возвращает:
            tuple:
                cent_mat (np.ndarray): матрица центроидов (C, D), C ≤ n_classes
                names_out (list[str]): имена классов, соответствующие строкам cent_mat
        """
        z = torch.nn.functional.normalize(embeddings_norm.float(), p=2, dim=1)
        lab = labels_long.long()
        counts = torch.bincount(lab, minlength=n_classes)

        centroid_rows: list[np.ndarray] = []
        names_out: list[str] = []

        cn = self.cfg.get('METRICS', 'class_names', [])
        cn = OmegaConf.to_container(cn, resolve=True) if cn else []
        if isinstance(cn, str):
            cn = [cn]
        cn = [str(x) for x in cn] if cn else []

        for c in range(n_classes):
            if int(counts[c].item()) < 1:
                continue
            mask = lab == c
            cent = z[mask].mean(dim=0)
            cent = torch.nn.functional.normalize(cent, p=2, dim=0)
            centroid_rows.append(cent.cpu().numpy())
            display = cn[c] if c < len(cn) else str(c)
            names_out.append(display)

        if not centroid_rows:
            return np.zeros((0, z.size(1))), []
        return np.stack(centroid_rows, axis=0), names_out

    def _plot_embedding_class_centroids_after_fit(self) -> None:
        """
        Строит 2D-визуализацию эмбеддингов и центроидов классов после обучения.

        Алгоритм:
            1. Прогоняет val_loader через модель → собирает все эмбеддинги
            2. Вычисляет центроиды классов
            3. Опционально сохраняет ``class_centroids.npy`` и ``class_names.txt``
            4. Строит 2D PCA-визуализацию через ``_plot_embedding_clusters_2d``

        Пропускает если:
            - ``plot_centroids_after_fit=False`` в конфиге
            - val_loader недоступен
            - num_classes < 2

        Сохраняет в:
            ``{runs_dir}/embeddings/{ModelClassName}/``
                - embedding_clusters.png
                - class_centroids.npy (если save_centroids_after_fit=True)
                - class_names.txt     (если save_centroids_after_fit=True)
        """
        if not self.cfg.get('EMBEDDING_METRICS', 'plot_centroids_after_fit', True):
            return
        loader = getattr(self, '_last_val_loader', None)
        if loader is None:
            logger.warning('No val loader for centroid plot')
            return

        n_classes = self._infer_num_classes()
        if n_classes is None or n_classes < 2:
            logger.info('Skipping centroid plot — need num_classes ≥ 2')
            return

        z, lab = self._embeddings_labels_from_loader(loader)
        cent_mat, names = self._embedding_class_centroids(z, lab, n_classes)
        if cent_mat.size == 0:
            return

        name = self.model.__class__.__name__
        sub = os.path.join(self.runs_dir, 'embeddings', name)
        os.makedirs(sub, exist_ok=True)
        if self.cfg.get('EMBEDDING_METRICS', 'save_centroids_after_fit', True):
            np.save(os.path.join(sub, 'class_centroids.npy'), cent_mat)
            with open(os.path.join(sub, 'class_names.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(names))
            logger.info(f'Class centroids saved → {sub}')

        clusters_path = os.path.join(sub, 'embedding_clusters.png')
        _plot_embedding_clusters_2d(
            z.numpy(),
            lab.numpy(),
            cent_mat,
            names,
            clusters_path,
            title=f'{name} — val embeddings and L2 class centroids (PCA2D)',
        )

    def fit(self):
        """
        Расширяет BaseTrainer.fit(): добавляет визуализацию центроидов после обучения.

        Порядок:
            1. ``super().fit()`` — полный цикл обучения (BaseTrainer)
            2. ``_plot_embedding_class_centroids_after_fit()``
        """
        super().fit()
        self._plot_embedding_class_centroids_after_fit()


ClassificationTrainer = TrainerClassification
EmbeddingTrainer = TrainerEmbeddings
EmbeddingsTrainer = TrainerEmbeddings
