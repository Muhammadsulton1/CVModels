import os
from collections import deque
from copy import deepcopy

import torch

from models_src.config_reader import ConfigReader
from utils.logger import setup_logger

logger = setup_logger()


class EarlyStopping:
    def __init__(self, model_name: str = 'model'):
        self.cfg = ConfigReader()
        self.model_name = model_name
        self.patience = int(self.cfg.get('EARLY_STOPPING', 'patience', 10))
        self.min_delta = float(self.cfg.get('EARLY_STOPPING', 'min_delta', 0.01))
        self.monitor = str(self.cfg.get('EARLY_STOPPING', 'monitor', 'val_loss'))
        self.mode = str(self.cfg.get('EARLY_STOPPING', 'mode', 'min')).lower()
        if self.mode not in ('min', 'max'):
            raise ValueError("EARLY_STOPPING.mode must be 'min' or 'max'")
        self.patience_counter = 0
        self.best_score = float('inf') if self.mode == 'min' else -float('inf')
        self.best_cmp_mode: str = self.mode
        self.best_epoch: int = -1
        self.best_metric_log_name: str = self.monitor
        self.best_val_loss = float('inf')
        self.stop = False
        self.best_nets: deque = deque(maxlen=int(self.cfg.get('EARLY_STOPPING', 'n_best_nets', 5)))

    def __str__(self):
        return (f'EarlyStopping(monitor={self.monitor}, mode={self.mode}, '
                f'patience={self.patience}, best_score={self.best_score:.4f})')

    @staticmethod
    def _is_improvement(value: float, best_score: float, comparison_mode: str, min_delta: float) -> bool:
        comparison_mode = str(comparison_mode).lower()
        if comparison_mode == 'min':
            return best_score - value > min_delta
        return value - best_score > min_delta

    def step(
            self,
            model,
            criterion,
            optimizer,
            epoch: int,
            monitor_value: float,
            val_loss: float | None = None,
            *,
            comparison_mode: str | None = None,
            metric_log_name: str | None = None,
    ):
        """comparison_mode — 'min' / 'max' для сравнения monitor_value (например, при fallback на val_loss нужен min даже если в конфиге max)."""
        cmp_mode = (comparison_mode or self.mode).lower()
        if cmp_mode not in ('min', 'max'):
            raise ValueError("comparison_mode must be 'min' or 'max'")

        monitor_value = float(monitor_value)

        if cmp_mode != self.best_cmp_mode:
            self.best_score = float('inf') if cmp_mode == 'min' else -float('inf')
            self.best_cmp_mode = cmp_mode
            self.patience_counter = 0
            self.best_nets.clear()
            self.best_epoch = -1
            self.best_metric_log_name = self.monitor

        if self._is_improvement(monitor_value, self.best_score, cmp_mode, self.min_delta):
            self.best_score = monitor_value
            self.best_epoch = int(epoch)
            if metric_log_name is not None:
                self.best_metric_log_name = str(metric_log_name)
            if val_loss is not None:
                self.best_val_loss = float(val_loss)
            self.patience_counter = 0
            self.stop = False
            self.best_nets.append(deepcopy(model.state_dict()))
            self.save_checkpoint(model, criterion, optimizer, epoch, monitor_value, val_loss)
        else:
            self.patience_counter += 1
            logger.debug(
                f'No improvement in {self.monitor}. Patience counter: {self.patience_counter}/{self.patience}'
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

    def save_checkpoint(self, model, criterion, optimizer, epoch: int, monitor_value: float, val_loss: float | None):
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
            'criterion_state_dict': criterion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'monitor': self.monitor,
            'monitor_value': monitor_value,
            'epoch': epoch
        }, path)
        if was_training:
            model.train()
        logger.info(
            f'Checkpoint saved → {path}  [epoch={epoch}, {self.monitor}={monitor_value:.4f}]'
        )
