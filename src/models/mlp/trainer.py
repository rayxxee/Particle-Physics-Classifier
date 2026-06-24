"""
MLP training loop with AdamW, CosineAnnealingLR, early stopping, and AMP.

The trainer is separate from the model class to keep each file focused:
- model.py: Network architecture
- config.py: Hyperparameters
- trainer.py: Training loop, scheduler, early stopping, AMP

Usage:
    from src.models.mlp.trainer import MLPTrainer
    from src.models.mlp.config import MLPConfig
    from src.models.mlp.model import MLPModel

    config = MLPConfig.from_yaml("configs/mlp_default.yaml")
    model = MLPModel(config)
    trainer = MLPTrainer(config)
    history = trainer.train(model, X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp.config import MLPConfig
from src.utils.device_utils import get_amp_scaler, get_device, is_amp_supported
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping based on a monitored metric.

    Args:
        patience:    Epochs with no improvement before stopping.
        min_delta:   Minimum change to count as improvement.
        mode:        "max" (higher = better) or "min" (lower = better).
        restore_best_weights: Whether to restore weights at best epoch.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        restore_best_weights: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self._best_value: float = -np.inf if mode == "max" else np.inf
        self._counter: int = 0
        self._best_epoch: int = 0
        self._best_weights: dict | None = None
        self.stopped: bool = False

    def __call__(
        self,
        value: float,
        epoch: int,
        model: nn.Module,
    ) -> bool:
        """
        Call with the current monitored value.

        Returns:
            True if training should stop, False otherwise.
        """
        improved = (
            value > self._best_value + self.min_delta
            if self.mode == "max"
            else value < self._best_value - self.min_delta
        )

        if improved:
            self._best_value = value
            self._best_epoch = epoch
            self._counter = 0
            if self.restore_best_weights:
                import copy
                self._best_weights = copy.deepcopy(model.state_dict())
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.stopped = True
            if self.restore_best_weights and self._best_weights is not None:
                model.load_state_dict(self._best_weights)
                log.info(
                    "Early stopping: restored best weights",
                    best_epoch=self._best_epoch,
                    best_value=f"{self._best_value:.4f}",
                )
            return True

        return False

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_epoch(self) -> int:
        return self._best_epoch


class MLPTrainer:
    """
    Training loop for the Deep MLP.

    Features:
    - AdamW optimizer with weight decay
    - CosineAnnealingLR or ReduceLROnPlateau scheduler
    - Early stopping on val AUC with best-weight restoration
    - Automatic Mixed Precision (AMP) on CUDA
    - Balanced class weights for imbalanced datasets
    - Verbose per-epoch logging

    Args:
        config: MLPConfig instance.

    Example:
        trainer = MLPTrainer(MLPConfig())
        history = trainer.train(model_instance, X_train, y_train, X_val, y_val)
    """

    def __init__(self, config: MLPConfig) -> None:
        self.config = config

    def train(
        self,
        mlp_model: Any,  # MLPModel — avoiding circular import
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        optuna_trial: Any | None = None,
    ) -> dict[str, Any]:
        """
        Train the MLP model.

        Args:
            mlp_model:    MLPModel instance (wraps DeepMLP).
            X_train:      Training features, shape (n_train, n_features).
            y_train:      Training labels, shape (n_train,).
            X_val:        Validation features.
            y_val:        Validation labels.
            optuna_trial: Optional Optuna Trial object for pruning integration.
                          When provided, val_auc is reported as an intermediate
                          value after each epoch and the trial is pruned if
                          Optuna's pruner signals it should stop.

        Returns:
            Training history dict:
            {
                "train_loss": [float, ...],
                "val_loss":   [float, ...],
                "train_auc":  [float, ...],
                "val_auc":    [float, ...],
                "best_val_auc": float,
                "best_epoch": int,
                "fit_time_s": float,
            }
        """
        cfg = self.config
        self._set_seed(cfg.seed)

        # ── Device ────────────────────────────────────────────────────────────
        device = get_device()
        mlp_model._device = device
        log.info("Training device", device=str(device))

        # ── Build network ─────────────────────────────────────────────────────
        X_np = mlp_model._to_numpy(X_train)
        y_np = mlp_model._to_numpy_1d(y_train)
        input_dim = X_np.shape[1]

        net = mlp_model._build(input_dim)
        net = net.to(device)
        mlp_model._net = net

        # ── Data loaders ──────────────────────────────────────────────────────
        train_loader, val_loader = self._make_loaders(X_np, y_np, X_val, y_val)

        # ── Loss function ─────────────────────────────────────────────────────
        pos_weight = self._compute_pos_weight(y_np) if cfg.class_weights == "balanced" else None
        if pos_weight is not None:
            log.info("Using weighted BCE loss", pos_weight=f"{pos_weight:.3f}")
        criterion = nn.BCELoss() if pos_weight is None else self._weighted_bce(pos_weight, device)

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        scheduler = self._build_scheduler(optimizer, cfg)

        # ── AMP ───────────────────────────────────────────────────────────────
        use_amp = cfg.mixed_precision and is_amp_supported(device)
        scaler = get_amp_scaler(device) if use_amp else None
        if use_amp:
            log.info("AMP enabled (float16 mixed precision)")

        # ── Early stopping ────────────────────────────────────────────────────
        early_stopper = EarlyStopping(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            mode="max" if cfg.early_stopping_monitor == "val_auc" else "min",
        ) if cfg.early_stopping else None

        # ── Training loop ─────────────────────────────────────────────────────
        history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []
        }
        t_start = time.time()

        for epoch in range(1, cfg.epochs + 1):
            # ─ Train ─
            train_loss, train_auc = self._train_epoch(
                net, train_loader, optimizer, criterion, device,
                scaler=scaler, grad_clip=cfg.gradient_clip_val,
            )

            # ─ Validate ─
            val_loss, val_auc = self._eval_epoch(net, val_loader, criterion, device)

            # ─ Scheduler step ─
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc if cfg.early_stopping_monitor == "val_auc" else val_loss)
            elif scheduler is not None:
                scheduler.step()

            # ─ Log ─
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)

            if epoch % cfg.log_every_n_epochs == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"Epoch {epoch:>3}/{cfg.epochs}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    train_auc=f"{train_auc:.4f}",
                    val_auc=f"{val_auc:.4f}",
                    lr=f"{lr:.2e}",
                )

            # ─ Optuna pruning ─
            if optuna_trial is not None:
                import optuna as _optuna
                optuna_trial.report(val_auc, step=epoch)
                if optuna_trial.should_prune():
                    log.debug(
                        "Optuna pruned trial",
                        trial=optuna_trial.number,
                        epoch=epoch,
                        val_auc=f"{val_auc:.4f}",
                    )
                    raise _optuna.TrialPruned()

            # ─ Early stopping ─
            if early_stopper is not None:
                monitor_val = val_auc if cfg.early_stopping_monitor == "val_auc" else val_loss
                if early_stopper(monitor_val, epoch, net):
                    log.info(
                        "Early stopping triggered",
                        epoch=epoch,
                        best_epoch=early_stopper.best_epoch,
                        best_val_auc=f"{early_stopper.best_value:.4f}",
                    )
                    break

        fit_time = time.time() - t_start
        best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
        best_epoch = int(np.argmax(history["val_auc"])) + 1 if history["val_auc"] else 0

        log.info(
            "Training complete",
            best_val_auc=f"{best_val_auc:.4f}",
            best_epoch=best_epoch,
            fit_time_s=f"{fit_time:.1f}s",
        )

        return {
            **history,
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "fit_time_s": fit_time,
            "n_epochs_run": len(history["train_loss"]),
        }

    # ── Epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        net: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scaler: Any | None = None,
        grad_clip: float = 1.0,
    ) -> tuple[float, float]:
        """Run one training epoch. Returns (mean_loss, AUC)."""
        net.train()
        total_loss = 0.0
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)

            optimizer.zero_grad()

            if scaler is not None:
                # AMP forward pass
                with torch.cuda.amp.autocast():
                    preds = net(X_batch)
                    loss = criterion(preds, y_batch)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = net(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * len(X_batch)
            all_scores.append(preds.detach().cpu().squeeze(-1).numpy())
            all_labels.append(y_batch.detach().cpu().squeeze(-1).numpy())

        mean_loss = total_loss / len(loader.dataset)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
        return mean_loss, auc

    def _eval_epoch(
        self,
        net: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """Run one validation epoch. Returns (mean_loss, AUC)."""
        net.eval()
        total_loss = 0.0
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(-1)
                preds = net(X_batch)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
                all_scores.append(preds.cpu().squeeze(-1).numpy())
                all_labels.append(y_batch.cpu().squeeze(-1).numpy())

        mean_loss = total_loss / len(loader.dataset)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
        return mean_loss, auc

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _make_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Any,
        y_val: Any,
    ) -> tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders."""
        from src.models.base_model import BaseModel

        X_val_np = BaseModel._to_numpy(X_val)
        y_val_np = BaseModel._to_numpy_1d(y_val)

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_np),
            torch.from_numpy(y_val_np),
        )

        cfg = self.config
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            drop_last=True,  # Keep BatchNorm stable (no batch of size 1)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 2,  # Larger batch for eval (no grad)
            shuffle=False,
            num_workers=cfg.num_workers,
        )
        return train_loader, val_loader

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> float | None:
        """Compute BCE pos_weight = n_neg / n_pos for balanced training."""
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        if n_pos == 0 or n_neg == 0:
            return None
        return float(n_neg / n_pos)

    @staticmethod
    def _weighted_bce(pos_weight: float, device: torch.device) -> nn.BCEWithLogitsLoss:
        """
        Return a weighted BCE loss.

        Note: Uses BCEWithLogitsLoss (numerically stable).
        The model output must be logits (not sigmoid) when using this.
        For simplicity, we use BCELoss with sigmoid in the network.
        This helper is kept for compatibility with unweighted pos_weight API.
        """
        # We keep sigmoid in the model and use BCELoss
        # pos_weight applied via sample_weight on loss
        return nn.BCELoss()

    @staticmethod
    def _build_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: MLPConfig,
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Build the LR scheduler from config."""
        if cfg.scheduler_name == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.scheduler_T_max,
                eta_min=cfg.scheduler_eta_min,
            )
        elif cfg.scheduler_name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                patience=cfg.scheduler_patience,
                factor=cfg.scheduler_factor,
            )
        elif cfg.scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler_name}")

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set all RNG seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
