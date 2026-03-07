import os
import time
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from omegaconf import OmegaConf
from copy import deepcopy
from collections import deque

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


class EarlyStopping:
    def __init__(self, model_name: str = 'model'):
        self.cfg = ConfigReader()
        self.model_name = model_name
        self.patience = int(self.cfg.get('EARLY_STOPPING', 'patience', 10))
        self.min_delta = float(self.cfg.get('EARLY_STOPPING', 'min_delta', 0.01))
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.stop = False
        self.best_nets: deque = deque(maxlen=int(self.cfg.get('EARLY_STOPPING', 'n_best_nets', 5)))

    def __str__(self):
        return (f'EarlyStopping(patience={self.patience}, '
                f'best_val_loss={self.best_val_loss:.4f})')

    def step(self, model, optimizer, epoch: int, val_loss: float):
        if self.best_val_loss - val_loss > self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.stop = False
            self.best_nets.append(deepcopy(model.state_dict()))
            self.save_checkpoint(model, optimizer, epoch)
        else:
            self.patience_counter += 1
            logger.debug(
                f'No improvement. Patience counter: {self.patience_counter}/{self.patience}'
            )
            if self.patience_counter >= self.patience:
                self.stop = True
                logger.info('Early stopping triggered')

    def avg_net(self):
        """Model Soup: усреднение весов лучших чекпоинтов."""
        if not self.best_nets:
            return None
        averaged = {}
        for key in self.best_nets[0].keys():
            orig_dtype = self.best_nets[0][key].dtype
            if orig_dtype in (torch.int32, torch.int64):
                averaged[key] = self.best_nets[-1][key]
            else:
                averaged[key] = (
                    torch.stack([sd[key].float() for sd in self.best_nets])
                    .mean(dim=0)
                    .to(orig_dtype)
                )
        return averaged

    def save_checkpoint(self, model, optimizer, epoch: int):
        _trainer_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_trainer_dir)
        save_dir = os.path.join(_project_root, 'weights', self.model_name)
        filename = 'best_checkpoint.pth'
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)

        was_training = model.training
        model.eval()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': self.best_val_loss,
            'epoch': epoch
        }, path)
        if was_training:
            model.train()
        logger.info(
            f'Checkpoint saved → {path}  [epoch={epoch}, val_loss={self.best_val_loss:.4f}]'
        )


class Trainer:
    def __init__(self, model, criterion, dataset):
        self.cfg = ConfigReader()
        self.dataset = dataset
        self.val_loss_history: list = []
        self.train_loss_history: list = []

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.writers = [SummaryWriter(log_dir=os.path.join(_project_root, 'runs'))]
        self.use_amp = bool(self.cfg.get('TRAINER', 'use_amp', True))
        self.scaler = torch.amp.GradScaler(device=self.device.type) if self.use_amp else None

        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        logger.info(f'Trainer initialized | device={self.device} | AMP={self.use_amp} | '
                    f'model={model.__class__.__name__}')

    def _get_dataset_targets(self):
        """
        Универсально получает метки для стратификации.
        Порядок поиска:
        1. Атрибуты из конфига DATASET.targets_attrs (по умолчанию: targets, labels, y)
        2. Для dict-датасетов: ключ из DATASET.label_key
        3. Для tuple/list элементов: второй элемент
        """
        base_dataset = self.dataset
        if isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset

        targets_attrs = self.cfg.get('DATASET', 'targets_attrs', ['targets', 'labels', 'y'])
        if targets_attrs is None:
            targets_attrs = ['targets', 'labels', 'y']
        if isinstance(targets_attrs, str):
            targets_attrs = [targets_attrs]

        for attr in targets_attrs:
            if hasattr(base_dataset, attr):
                arr = getattr(base_dataset, attr)
                if isinstance(self.dataset, Subset):
                    return np.array(arr)[self.dataset.indices] if hasattr(arr, '__getitem__') else arr
                return arr

        if hasattr(base_dataset, 'data'):
            data = base_dataset.data
            if isinstance(data, (list, tuple)) and data:
                labels = [self._get_label_from_sample(s) for s in data]
                class_mapping = self._get_class_mapping(base_dataset)
                if class_mapping:
                    labels = [class_mapping.get(lbl, lbl) for lbl in labels]
                if isinstance(self.dataset, Subset):
                    return np.array(labels)[self.dataset.indices]
                return labels

        return None

    def _get_class_mapping(self, dataset):
        """Получает маппинг имён классов в индексы из датасета."""
        class_mapping_attr = self.cfg.get('DATASET', 'class_mapping_attr', None)

        if class_mapping_attr and hasattr(dataset, class_mapping_attr):
            return getattr(dataset, class_mapping_attr)

        for attr in ['class_to_idx', 'classes_to_idx', 'label_to_idx']:
            if hasattr(dataset, attr):
                return getattr(dataset, attr)

        return None

    def _get_label_from_sample(self, sample):
        """
        Извлекает метку из элемента датасета.
        Ключ для dict берётся из конфига DATASET.label_key.
        """
        label_key = self.cfg.get('DATASET', 'label_key', None)

        if isinstance(sample, dict):
            if label_key and label_key in sample:
                val = sample[label_key]
                if isinstance(val, (list, tuple)):
                    from collections import Counter
                    return Counter(val).most_common()[0][0]
                return val
            for key in ['label', 'target', 'class', 'y']:
                if key in sample:
                    val = sample[key]
                    if isinstance(val, (list, tuple)):
                        from collections import Counter
                        return Counter(val).most_common()[0][0]
                    return val

        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            return sample[1]

        return 0

    def _get_num_classes(self, dataset=None) -> int | None:
        """
        Универсально определяет число классов: из конфига (DATASET.num_classes),
        из датасета (classes, class_to_idx) или из меток.
        """
        cfg_num = self.cfg.get('DATASET', 'num_classes', None)
        if cfg_num is not None and cfg_num != 'null':
            try:
                return int(cfg_num)
            except (TypeError, ValueError):
                pass

        base = dataset if dataset is not None else self.dataset
        if isinstance(base, Subset):
            base = base.dataset

        if hasattr(base, 'classes') and base.classes:
            return len(base.classes)
        if hasattr(base, 'class_to_idx') and base.class_to_idx:
            return len(base.class_to_idx)

        targets = self._get_dataset_targets()
        if targets is not None:
            arr = np.asarray(targets)
            if len(arr) > 0:
                return int(arr.max()) + 1
        return None

    def _compute_class_weights(self, dataset) -> torch.Tensor | None:
        """
        Вычисляет веса классов для CrossEntropyLoss (inverse frequency).
        Универсально работает с Dataset, Subset, ImageFolder, QualityDataset и др.
        """
        all_targets = self._get_dataset_targets()
        if all_targets is None:
            return None

        all_targets = np.array(all_targets) if not isinstance(all_targets, np.ndarray) else all_targets
        targets = all_targets

        if isinstance(dataset, Subset):
            indices = dataset.indices
            targets = all_targets[indices]

        if len(targets) == 0:
            return None

        n_classes = self._get_num_classes(dataset)
        if n_classes is None or n_classes == 0:
            n_classes = int(all_targets.max()) + 1 if len(all_targets) > 0 else 0
        if n_classes == 0:
            return None

        unique, counts = np.unique(targets, return_counts=True)

        total = len(targets)
        weights = np.ones(n_classes, dtype=np.float32)
        for cls, cnt in zip(unique, counts):
            weights[int(cls)] = total / (n_classes * max(int(cnt), 1))

        weights = torch.tensor(weights, dtype=torch.float32)
        logger.info(f'Class weights computed (n_classes={n_classes}): {[f"{w:.3f}" for w in weights.tolist()]}')
        return weights

    def prepare_data(self):
        train_proc = float(self.cfg.get('TRAINER', 'train_proc', 0.75))
        batch_size = int(self.cfg.get('TRAINER', 'batch_size', 64))
        num_workers = int(self.cfg.get('TRAINER', 'num_workers', 8))
        shuffle = bool(self.cfg.get('TRAINER', 'shuffle', True))
        split_strategy = str(self.cfg.get('TRAINER', 'split_strategy', 'random')).lower()

        n_total = len(self.dataset)
        if n_total < 2:
            raise ValueError(f'Dataset too small for split: {n_total} samples')

        train_size = int(n_total * train_proc)
        val_size = n_total - train_size
        if train_size < 1 or val_size < 1:
            raise ValueError(
                f'Invalid split: train_proc={train_proc} gives train={train_size}, val={val_size}. '
                f'Adjust train_proc or use more data.'
            )

        train_dataset = val_dataset = None

        if split_strategy == 'sequential':
            indices = list(range(n_total))
            split_idx = train_size
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]
            train_dataset = Subset(self.dataset, train_indices)
            val_dataset = Subset(self.dataset, val_indices)
        elif split_strategy == 'stratified':
            targets = self._get_dataset_targets()
            if targets is None:
                logger.warning('Stratified split requested but dataset has no targets. Falling back to random.')
                split_strategy = 'random'
            else:
                try:
                    targets = np.array(targets) if not isinstance(targets, np.ndarray) else targets
                    train_indices, val_indices = train_test_split(
                        np.arange(n_total), train_size=train_proc, stratify=targets, random_state=42
                    )
                    train_dataset = Subset(self.dataset, train_indices.tolist())
                    val_dataset = Subset(self.dataset, val_indices.tolist())
                except ValueError as e:
                    logger.warning(f'Stratified split failed ({e}). Falling back to random.')
                    split_strategy = 'random'

        if split_strategy == 'random' or train_dataset is None:
            train_dataset, val_dataset = random_split(
                self.dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        use_class_weights = self.cfg.get('TRAINER', 'use_class_weights', False)
        if use_class_weights:
            weights = self._compute_class_weights(train_dataset)
            if weights is not None:
                weights = weights.to(self.device)
                self.criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
                self.criterion = self.criterion.to(self.device)
                logger.info('CrossEntropyLoss updated with class weights')
            else:
                logger.warning('use_class_weights=True but could not compute weights. Using default criterion.')

        pin = self.device.type == 'cuda'
        use_persistent = num_workers > 0

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin, persistent_workers=use_persistent)

        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=pin, persistent_workers=use_persistent)

        logger.info(f'Data split | strategy={split_strategy} | train={len(train_dataset)}, '
                    f'val={len(val_dataset)} | batch_size={batch_size}')

        return train_loader, val_loader

    def init_optimizer(self, model, lr):
        optimizer_type = str(self.cfg.get('TRAINER', 'optimizer_type', 'AdamW')).lower()
        weight_decay = float(self.cfg.get('TRAINER', 'weight_decay', 0.05))

        decay_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in ['bias', 'norm'])]
        no_decay_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and any(nd in n for nd in ['bias', 'norm'])]

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
        return optimizer.param_groups[0]['lr']

    def init_scheduler(self, optimizer, lr, num_epochs, scheduler_type='cosine'):
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
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(num_epochs - warmup_epochs), eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        elif s == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=eta_min
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
                optimizer, factor=plateau_factor,
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
        SINCE = time.time()

        self.model.train()
        total_loss = 0.0
        for x, y in tqdm(loader, desc='Training', leave=False):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            def _forward():
                if x.dim() == 5:
                    b, t, c, h, w = x.shape
                    x_ = x.view(b * t, c, h, w)
                else:
                    x_ = x
                prediction = self.model(x_)
                if prediction.size(0) != y.size(0):
                    prediction = prediction.view(y.size(0), -1, prediction.size(-1)).mean(dim=1)
                return self.criterion(prediction, y)

            if self.use_amp:
                with torch.autocast(device_type=self.device.type):
                    loss = _forward()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss = _forward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()

        time_elapsed = time.time() - SINCE

        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return total_loss / len(loader), time_elapsed

    def _get_metrics_config(self):
        """Читает настройки метрик из конфига."""
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
        """Вычисляет запрошенные метрики."""
        result = {}
        for name in metrics_list:
            if name == 'confusion_matrix':
                cm = confusion_matrix(y_true, y_pred)
                result[name] = cm
            elif name in METRIC_FUNCS:
                result[name] = float(METRIC_FUNCS[name](y_true, y_pred))
        return result

    def valid_epoch(self, loader) -> tuple:
        """Возвращает (val_loss, metrics_dict или None)."""
        SINCE = time.time()
        metrics_enabled, metrics_list, _ = self._get_metrics_config()
        all_preds, all_labels = [], []

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(loader, desc='Validation', leave=False):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                def _val_forward():
                    if x.dim() == 5:
                        b, t, c, h, w = x.shape
                        x_ = x.view(b * t, c, h, w)
                    else:
                        x_ = x
                    prediction = self.model(x_)
                    if prediction.size(0) != y.size(0):
                        prediction = prediction.view(y.size(0), -1, prediction.size(-1)).mean(dim=1)
                    return self.criterion(prediction, y), prediction

                if self.use_amp:
                    with torch.autocast(device_type=self.device.type):
                        loss, prediction = _val_forward()
                else:
                    loss, prediction = _val_forward()

                total_loss += loss.item()

                if metrics_enabled and metrics_list:
                    pred_cls = prediction.argmax(dim=1).cpu().numpy()
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

        logger.info('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return val_loss, metrics_dict, time_elapsed

    def _format_epoch_output(self, epoch: int, num_epochs: int, train_loss: float, val_loss: float,
                             lr: float, metrics: dict | None, train_time: float, val_time: float) -> str:
        """Форматирует красивый вывод эпохи."""
        _, _, class_names = self._get_metrics_config()
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
            metric_parts = [f'{k}: {v:.4f}' for k, v in metrics.items() if k != 'confusion_matrix']
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

            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                names = class_names[:cm.shape[0]] if class_names else [str(i) for i in range(cm.shape[0])]
                lines.append('╠' + '═' * (width - 2) + '╣')
                lines.append(box_line('Confusion Matrix:'))
                header = '  ' + '  '.join(f'{n:>8}' for n in names)
                lines.append(box_line(header))
                for i, row in enumerate(cm):
                    row_str = f'{names[i]:>6}: ' + '  '.join(f'{int(x):>8}' for x in row)
                    lines.append(box_line(row_str))

        lines.append('╚' + '═' * (width - 2) + '╝')
        return '\n'.join(lines)

    def load_checkpoint(self, mode: str = 'train'):
        checkpoint_path = self.cfg.get('TRAINER', 'load_checkpoint', 'weights/best.pth')

        if not checkpoint_path.endswith('.pth'):
            raise ValueError('Checkpoint must be a .pth file')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.train() if mode == 'train' else self.model.eval()
        logger.info(f'Checkpoint loaded: {checkpoint_path} '
                    f'[epoch={checkpoint["epoch"]}, mode={mode}]')
        return checkpoint['optimizer_state_dict'], checkpoint['epoch']

    def apply_model_soup(self, early_stopping: EarlyStopping):
        if self.cfg.get('EARLY_STOPPING', 'soup_strategy', 'avg') == 'avg':
            averaged = early_stopping.avg_net()
            if averaged is not None:
                self.model.load_state_dict(averaged)
                logger.info(
                    f'Model Soup applied from {len(early_stopping.best_nets)} checkpoints'
                )
        else:
            raise ValueError('Early stopping strategy must be "avg" another strategy are not implemented yet"')

    def _run_loop(self, optimizer, scheduler, train_loader, val_loader,
                  start_epoch: int, num_epochs: int,
                  early_stopping: EarlyStopping):
        try:
            for epoch in range(start_epoch, num_epochs):
                train_loss, train_time = self.train_epoch(optimizer, train_loader)
                val_loss, metrics_dict, val_time = self.valid_epoch(val_loader)
                current_lr = self.get_rate(optimizer)

                self.train_loss_history.append(train_loss)
                self.val_loss_history.append(val_loss)
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

                early_stopping.step(self.model, optimizer, epoch,
                                    val_loss)
                if early_stopping.stop:
                    break
        finally:
            for w in self.writers:
                w.close()
            logger.info('TensorBoard writer closed')

        self.apply_model_soup(early_stopping)

        if self.val_loss_history:
            best_idx = np.argmin(self.val_loss_history)
            best_epoch_num = start_epoch + best_idx
            best_val = min(self.val_loss_history)
            w = 50
            line = f'  Best val_loss: {best_val:.4f} at epoch {best_epoch_num}  '
            box = '\n'.join([
                '+' + '-' * w + '+',
                '|' + ' Training completed '.center(w) + '|',
                '+' + '-' * w + '+',
                '|' + line.ljust(w) + '|',
                '+' + '-' * w + '+',
            ])
            print('\n' + box + '\n')

    def fit(self):
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
        train_loader, val_loader = self.prepare_data()

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
