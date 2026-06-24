"""
GNN training loop with AdamW, CosineAnnealingLR, EarlyStopping, and AMP.

Mirrors the MLPTrainer structure:
  - AdamW optimizer
  - Cosine annealing LR scheduler
  - Early stopping on val AUC with best-weight restoration
  - AMP mixed precision on CUDA
  - Per-epoch logging

All torch / torch_geometric imports are LAZY (inside functions) to prevent
pytest collection hangs on Windows machines without torch_geometric installed.

Usage:
    from src.models.gnn.trainer import GNNTrainer
    from src.models.gnn.config import GNNConfig

    trainer = GNNTrainer(GNNConfig())
    history = trainer.train(gnn_model, X_train, y_train, X_val, y_val)
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import pandas as pd

from src.models.gnn.config import GNNConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)


class GNNTrainer:
    """
    Training loop for the EdgeConv GNN.

    Args:
        config: GNNConfig instance.
    """

    def __init__(self, config: GNNConfig) -> None:
        self.config = config

    def train(
        self,
        gnn_model: Any,  # GNNModel — avoiding circular import
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """
        Train the GNN.

        Args:
            gnn_model: GNNModel instance.
            X_train:   Training features, shape (n, n_features) or (n, n_p, n_f).
            y_train:   Training labels, shape (n,).
            X_val:     Validation features.
            y_val:     Validation labels.

        Returns:
            Training history dict with keys:
            train_loss, val_loss, train_auc, val_auc,
            best_val_auc, best_epoch, fit_time_s, n_epochs_run.
        """
        import torch
        import torch.nn as nn
        from sklearn.metrics import roc_auc_score
        from torch_geometric.loader import DataLoader

        from src.models.base_model import BaseModel
        from src.models.gnn.model import EdgeConvNet
        from src.utils.device_utils import get_device, is_amp_supported

        cfg = self.config
        self._set_seed(cfg.seed)

        # ── Device ────────────────────────────────────────────────────────────
        device = get_device()
        gnn_model._device = device
        log.info("GNN training device", device=str(device))

        # ── Prepare data ──────────────────────────────────────────────────────
        X_tr_np = gnn_model._to_numpy(X_train) if X_train.ndim != 3 else np.asarray(X_train, dtype=np.float32)
        y_tr_np = gnn_model._to_numpy_1d(y_train)
        X_v_np = gnn_model._to_numpy(X_val) if X_val.ndim != 3 else np.asarray(X_val, dtype=np.float32)
        y_v_np = gnn_model._to_numpy_1d(y_val)

        # Reshape flat → 3D
        X_tr_3d = gnn_model._reshape_input(X_tr_np)
        X_v_3d = gnn_model._reshape_input(X_v_np)

        # ── Build PyG datasets ────────────────────────────────────────────────
        train_dataset = self._build_pyg_dataset(X_tr_3d, y_tr_np, cfg.k_neighbors)
        val_dataset = self._build_pyg_dataset(X_v_3d, y_v_np, cfg.k_neighbors)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        # ── Build model ───────────────────────────────────────────────────────
        net = EdgeConvNet.build(cfg)
        net = net.to(device)
        gnn_model._net = net

        log.info(
            "GNN built",
            n_parameters=f"{net.n_parameters():,}",
            n_edge_conv_layers=cfg.n_edge_conv_layers,
            hidden_dim=cfg.hidden_dim,
        )

        # ── Optimizer + Scheduler ─────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.learning_rate * 0.01,
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        criterion = nn.BCELoss()

        # ── AMP ───────────────────────────────────────────────────────────────
        use_amp = cfg.mixed_precision and is_amp_supported(device)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # ── Early stopping state ──────────────────────────────────────────────
        best_val_auc = 0.0
        best_epoch = 0
        best_weights: dict | None = None
        no_improve = 0

        history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []
        }
        t_start = time.time()

        for epoch in range(1, cfg.epochs + 1):
            # ─ Train epoch ─
            train_loss, train_auc = self._train_epoch(
                net, train_loader, optimizer, criterion, device,
                scaler=scaler, grad_clip=cfg.gradient_clip_val,
            )

            # ─ Val epoch ─
            val_loss, val_auc = self._eval_epoch(net, val_loader, criterion, device)

            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)

            if epoch % cfg.log_every_n_epochs == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"GNN Epoch {epoch:>3}/{cfg.epochs}",
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
                            "GNN early stopping triggered",
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
            "GNN training complete",
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
        """Run one training epoch on graph batches. Returns (mean_loss, AUC)."""
        import torch
        from sklearn.metrics import roc_auc_score

        net.train()
        total_loss = 0.0
        all_scores: list = []
        all_labels: list = []
        n_total = 0

        for batch in loader:
            batch = batch.to(device)
            y = batch.y.float().unsqueeze(-1)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = net(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(preds, y)
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = net(batch.x, batch.edge_index, batch.batch)
                loss = criterion(preds, y)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                optimizer.step()

            n = len(y)
            total_loss += loss.item() * n
            n_total += n
            all_scores.append(preds.detach().cpu().squeeze(-1).numpy())
            all_labels.append(y.cpu().squeeze(-1).numpy())

        mean_loss = total_loss / max(n_total, 1)
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
        n_total = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                y = batch.y.float().unsqueeze(-1)
                preds = net(batch.x, batch.edge_index, batch.batch)
                loss = criterion(preds, y)
                n = len(y)
                total_loss += loss.item() * n
                n_total += n
                all_scores.append(preds.cpu().squeeze(-1).numpy())
                all_labels.append(y.cpu().squeeze(-1).numpy())

        mean_loss = total_loss / max(n_total, 1)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
        return mean_loss, auc

    # ── Data construction ─────────────────────────────────────────────────────

    def _build_pyg_dataset(
        self,
        X_3d: np.ndarray,
        y: np.ndarray,
        k: int,
    ) -> list:
        """
        Build a list of PyG Data objects (one per event).

        Edges are built per event using kNN in the node feature space.
        For HIGGS, uses (eta, phi) = features at index 1, 2.
        """
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import knn_graph

        dataset = []
        n_events = len(X_3d)

        for i in range(n_events):
            x_np = X_3d[i]  # (n_particles, n_node_features)
            x_t = torch.tensor(x_np, dtype=torch.float32)

            # kNN graph on (eta, phi) or all features
            if x_t.shape[1] >= 3:
                pos = x_t[:, 1:3]
            else:
                pos = x_t

            # Build single-graph knn (no batch dimension)
            edge_index = knn_graph(pos, k=min(k, x_t.shape[0] - 1), loop=False)

            label = torch.tensor([y[i]], dtype=torch.float32)
            dataset.append(Data(x=x_t, edge_index=edge_index, y=label))

        return dataset

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
