"""
Particle Transformer training loop.

Features:
  - AdamW optimizer with linear warmup + cosine LR schedule
  - Early stopping on val AUC with best-weight restoration
  - AMP mixed precision on CUDA
  - Standard PyTorch DataLoader (no external deps beyond torch)

All torch imports are LAZY (inside functions) to prevent pytest
collection hangs on Windows machines.

Usage:
    from src.models.transformer.trainer import TransformerTrainer
    from src.models.transformer.config import TransformerConfig

    trainer = TransformerTrainer(TransformerConfig())
    history = trainer.train(transformer_model, X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd

from src.models.transformer.config import TransformerConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class TransformerTrainer:
    """
    Training loop for the Particle Transformer.

    Args:
        config: TransformerConfig instance.
    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config

    def train(
        self,
        transformer_model: Any,  # TransformerModel — avoiding circular import
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """
        Train the Particle Transformer.

        Args:
            transformer_model: TransformerModel instance.
            X_train, y_train:  Training data.
            X_val, y_val:      Validation data.

        Returns:
            Training history dict.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from src.models.transformer.model import ParticleTransformerNet
        from src.utils.device_utils import get_device, is_amp_supported

        cfg = self.config
        self._set_seed(cfg.seed)

        # ── Device ────────────────────────────────────────────────────────────
        device = get_device()
        transformer_model._device = device
        log.info("Transformer training device", device=str(device))

        # ── Prepare data ──────────────────────────────────────────────────────
        X_tr_np = transformer_model._to_numpy(X_train)
        y_tr_np = transformer_model._to_numpy_1d(y_train)
        X_v_np = transformer_model._to_numpy(X_val)
        y_v_np = transformer_model._to_numpy_1d(y_val)

        # Reshape flat → (n_events, n_particles, n_features)
        X_tr_3d = transformer_model._reshape_input(X_tr_np)
        X_v_3d = transformer_model._reshape_input(X_v_np)

        # ── DataLoaders ───────────────────────────────────────────────────────
        train_ds = TensorDataset(
            torch.from_numpy(X_tr_3d),
            torch.from_numpy(y_tr_np),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_v_3d),
            torch.from_numpy(y_v_np),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        # ── Build model ───────────────────────────────────────────────────────
        net = ParticleTransformerNet.build(cfg)
        net = net.to(device)
        transformer_model._net = net

        log.info(
            "Particle Transformer built",
            n_parameters=f"{net.n_parameters():,}",
            architecture=net.architecture_str(),
        )

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Warmup + cosine schedule
        total_steps = cfg.epochs
        warmup_steps = cfg.warmup_epochs

        def lr_lambda(epoch: int) -> float:
            """Linear warmup then cosine decay."""
            if epoch < warmup_steps:
                return float(epoch + 1) / float(max(warmup_steps, 1))
            progress = (epoch - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # ── Loss ──────────────────────────────────────────────────────────────
        criterion = nn.BCELoss()

        # ── AMP ───────────────────────────────────────────────────────────────
        use_amp = cfg.mixed_precision and is_amp_supported(device)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            log.info("AMP enabled (float16 mixed precision)")

        # ── Early stopping ─────────────────────────────────────────────────────
        best_val_auc = 0.0
        best_epoch = 0
        best_weights: dict | None = None
        no_improve = 0

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

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)

            if epoch % cfg.log_every_n_epochs == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"Transformer Epoch {epoch:>3}/{cfg.epochs}",
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    train_auc=f"{train_auc:.4f}",
                    val_auc=f"{val_auc:.4f}",
                    lr=f"{lr:.2e}",
                )

            # ─ Early stopping ─
            if cfg.early_stopping:
                if val_auc > best_val_auc + cfg.early_stopping_min_delta:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    no_improve = 0
                    best_weights = copy.deepcopy(net.state_dict())
                else:
                    no_improve += 1
                    if no_improve >= cfg.early_stopping_patience:
                        log.info(
                            "Transformer early stopping triggered",
                            epoch=epoch,
                            best_epoch=best_epoch,
                            best_val_auc=f"{best_val_auc:.4f}",
                        )
                        if best_weights is not None:
                            net.load_state_dict(best_weights)
                        break
            else:
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch

        if best_val_auc == 0.0:
            best_val_auc = max(history["val_auc"]) if history["val_auc"] else 0.0
            best_epoch = int(np.argmax(history["val_auc"])) + 1 if history["val_auc"] else 0

        fit_time = time.time() - t_start

        log.info(
            "Transformer training complete",
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
        net: Any,
        loader: Any,
        optimizer: Any,
        criterion: Any,
        device: Any,
        scaler: Any = None,
        grad_clip: float = 1.0,
    ) -> tuple[float, float]:
        """Run one training epoch. Returns (mean_loss, AUC)."""
        import torch
        from sklearn.metrics import roc_auc_score

        net.train()
        total_loss = 0.0
        all_scores: list = []
        all_labels: list = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)

            optimizer.zero_grad()

            if scaler is not None:
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
            all_labels.append(y_batch.cpu().squeeze(-1).numpy())

        mean_loss = total_loss / max(len(loader.dataset), 1)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
        return mean_loss, auc

    def _eval_epoch(
        self,
        net: Any,
        loader: Any,
        criterion: Any,
        device: Any,
    ) -> tuple[float, float]:
        """Run one validation epoch. Returns (mean_loss, AUC)."""
        import torch
        from sklearn.metrics import roc_auc_score

        net.eval()
        total_loss = 0.0
        all_scores: list = []
        all_labels: list = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(-1)
                preds = net(X_batch)
                loss = criterion(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
                all_scores.append(preds.cpu().squeeze(-1).numpy())
                all_labels.append(y_batch.cpu().squeeze(-1).numpy())

        mean_loss = total_loss / max(len(loader.dataset), 1)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
        return mean_loss, auc

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
