"""
EdgeConv GNN for particle physics event classification.

Architecture (DGCNN-style):
    Each event = graph where nodes are final-state particles.

    Input:
        X shape (n_events, n_particles, n_node_features)
        For HIGGS: (N, 4, 7)  — 28 flat features reshaped to 4 particles × 7 features

    Graph construction:
        k-nearest neighbors (k=8) in (eta, phi) = (feature_1, feature_2) space

    Message passing:
        EdgeConv × 3 layers (each layer updates node features using edge MLPs)
        EdgeConv: h_i = max_{j ∈ kNN(i)} MLP([h_i, h_j - h_i])

    Pooling:
        Global mean pooling → single graph embedding

    Head:
        MLP(128 → 64) → Linear(1) → Sigmoid

All torch_geometric imports are LAZY (inside functions, not module level).
Reason: pytest hangs on collection if torch_geometric imported at module top
level on this Windows machine.

Usage:
    from src.models.gnn.config import GNNConfig
    from src.models.gnn.model import GNNModel

    config = GNNConfig(k_neighbors=8, n_edge_conv_layers=3, hidden_dim=64)
    model = GNNModel(config)
    best_auc = model.fit(X_train, y_train, X_val, y_val)
    # X shape: (n_events, 28) — reshaped internally to (n_events, 4, 7)
    scores = model.predict_proba(X_test)  # shape (n_test,)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base_model import BaseModel
from src.models.gnn.config import GNNConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)


# ─── PyTorch module (lazy imports inside class) ───────────────────────────────

class EdgeConvNet:
    """
    Wrapper around the PyTorch Geometric EdgeConv network.
    Instantiated lazily to avoid top-level torch_geometric import.
    """

    @staticmethod
    def build(config: GNNConfig):
        """Build the EdgeConv network. All PyG imports are inside this function."""
        import torch
        import torch.nn as nn

        try:
            from torch_geometric.nn import EdgeConv, global_mean_pool
        except ImportError as e:
            raise ImportError(
                "torch_geometric is required for the GNN model.\n"
                "Install with: pip install torch_geometric\n"
                "Or in Colab: !pip install torch_geometric"
            ) from e

        class _EdgeConvGNN(nn.Module):
            """
            DGCNN-style EdgeConv GNN.

            Nodes: particles. Edges: k-nearest neighbors.
            Message: edge MLP over [h_i || h_j - h_i].
            Readout: global mean pooling → MLP head.
            """

            def __init__(self, cfg: GNNConfig) -> None:
                super().__init__()
                self.cfg = cfg

                # ── EdgeConv layers ───────────────────────────────────────────
                self.edge_convs = nn.ModuleList()
                in_dim = cfg.n_node_features

                for layer_idx in range(cfg.n_edge_conv_layers):
                    out_dim = cfg.hidden_dim
                    # EdgeConv MLP takes [h_i || h_j - h_i] → 2 * in_dim input
                    edge_mlp = nn.Sequential(
                        nn.Linear(2 * in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(out_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU(inplace=True),
                    )
                    self.edge_convs.append(EdgeConv(edge_mlp, aggr="max"))
                    in_dim = out_dim

                # ── Readout MLP head ──────────────────────────────────────────
                head_layers: list[nn.Module] = []
                h_in = in_dim
                for h_dim in cfg.mlp_head_dims:
                    head_layers.extend([
                        nn.Linear(h_in, h_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=cfg.dropout),
                    ])
                    h_in = h_dim
                head_layers.extend([
                    nn.Linear(h_in, 1),
                    nn.Sigmoid(),
                ])
                self.head = nn.Sequential(*head_layers)

            def forward(self, x: Any, edge_index: Any, batch: Any) -> Any:
                """
                Forward pass.

                Args:
                    x:          Node features (n_total_nodes, n_node_features).
                    edge_index: Graph connectivity (2, n_edges).
                    batch:      Batch assignment vector (n_total_nodes,).

                Returns:
                    Tensor shape (batch_size, 1), values in [0, 1].
                """
                # Message passing
                for edge_conv in self.edge_convs:
                    x = edge_conv(x, edge_index)

                # Global mean pooling
                pooled = global_mean_pool(x, batch)

                # Classification head
                return self.head(pooled)

            def n_parameters(self) -> int:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)

        return _EdgeConvGNN(config)

    @staticmethod
    def build_knn_graph(x: Any, k: int, batch: Any):
        """
        Build a k-nearest neighbor graph in feature space.

        Uses (eta, phi) = (feature[1], feature[2]) for physics-motivated
        graph construction, but falls back to all features if n_features < 3.

        Args:
            x:     Node features, shape (n_total_nodes, n_features).
            k:     Number of nearest neighbors.
            batch: Batch vector.

        Returns:
            edge_index: (2, n_edges)
        """
        try:
            from torch_geometric.nn import knn_graph
        except ImportError:
            from torch_geometric.nn import knn_graph  # type: ignore

        import torch

        # Use (eta, phi) = features at index 1, 2 for spatial proximity
        # This mimics HEP-motivated ΔR = sqrt(Δη² + Δφ²) connectivity
        if x.shape[1] >= 3:
            pos = x[:, 1:3]  # (eta, phi) slice
        else:
            pos = x  # fallback to all features

        # knn_graph returns edges within each event's subgraph (batch-aware)
        return knn_graph(pos, k=k, batch=batch, loop=False)


# ─── GNNModel (BaseModel interface) ──────────────────────────────────────────

class GNNModel(BaseModel):
    """
    EdgeConv GNN for particle physics event classification.

    Wraps the PyTorch Geometric EdgeConv network in the BaseModel interface.
    Input X can be:
      - (n_events, n_features): flat → reshaped to (n_events, n_particles, n_node_features)
      - (n_events, n_particles, n_node_features): already 3D

    For HIGGS dataset: 28 flat features → 4 particles × 7 features.

    Args:
        config: GNNConfig. Uses defaults if None.
    """

    def __init__(self, config: GNNConfig | None = None) -> None:
        self.config = config or GNNConfig()
        super().__init__(model_name="gnn")
        self._net: Any = None
        self._device: Any = None

    def _reshape_input(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape flat input to 3D (n_events, n_particles, n_node_features).

        If X is already 3D, returns as-is.
        If X is 2D flat: shape (n_events, n_features), reshapes to
        (n_events, n_particles, n_node_features).
        """
        if X.ndim == 3:
            return X
        n_events = X.shape[0]
        n_p = self.config.n_particles
        n_f = self.config.n_node_features
        expected = n_p * n_f
        if X.shape[1] != expected:
            # Silently adapt: truncate or pad to expected length
            if X.shape[1] > expected:
                X = X[:, :expected]
            else:
                pad = np.zeros((n_events, expected - X.shape[1]), dtype=X.dtype)
                X = np.hstack([X, pad])
        return X.reshape(n_events, n_p, n_f)

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        **kwargs: Any,
    ) -> float:
        """
        Train the GNN.

        Args:
            X_train, y_train: Training data. X shape (n, n_features) or (n, n_p, n_f).
            X_val, y_val:     Validation data.
            **kwargs:         Ignored (for interface compatibility).

        Returns:
            best_val_auc (float)
        """
        from src.models.gnn.trainer import GNNTrainer

        trainer = GNNTrainer(config=self.config)
        history = trainer.train(self, X_train, y_train, X_val, y_val)

        self._is_fitted = True
        self._fit_time_s = history.get("fit_time_s")
        self._metadata.update({
            "best_val_auc": history.get("best_val_auc"),
            "best_epoch": history.get("best_epoch"),
            "n_parameters": self._net.n_parameters() if self._net else None,
        })

        return history.get("best_val_auc", 0.0)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return P(signal) for each event.

        Args:
            X: Features, shape (n_events, n_features) or (n_events, n_p, n_f).

        Returns:
            1D float32 array of shape (n_events,), values in [0, 1].
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("GNNModel has not been fitted. Call .fit() first.")

        import torch

        X_np = self._to_numpy(X) if X.ndim != 3 else np.asarray(X, dtype=np.float32)
        X_3d = self._reshape_input(X_np)

        self._net.eval()
        all_scores = []

        # Batch through events (build graphs one batch at a time)
        bs = self.config.batch_size * 2
        for i in range(0, len(X_3d), bs):
            X_batch = X_3d[i : i + bs]
            batch_scores = self._predict_batch(X_batch)
            all_scores.append(batch_scores)

        return np.concatenate(all_scores, axis=0)

    def _predict_batch(self, X_3d: np.ndarray) -> np.ndarray:
        """Run inference on a 3D batch (n_events, n_particles, n_node_features)."""
        import torch

        data = self._build_pyg_batch(X_3d)
        data = data.to(self._device)

        with torch.no_grad():
            scores = self._net(data.x, data.edge_index, data.batch)

        return scores.squeeze(-1).cpu().numpy().astype(np.float32)

    def _build_pyg_batch(self, X_3d: np.ndarray) -> Any:
        """
        Convert a numpy 3D array to a PyTorch Geometric Batch object.

        Each event becomes one graph in the batch.
        """
        import torch
        from torch_geometric.data import Batch, Data

        graphs = []
        for event in X_3d:
            # event shape: (n_particles, n_node_features)
            x_t = torch.tensor(event, dtype=torch.float32)
            graphs.append(Data(x=x_t))

        # Collate into batch
        batch = Batch.from_data_list(graphs)

        # Build kNN graph on the batched node features
        edge_index = EdgeConvNet.build_knn_graph(
            batch.x, k=self.config.k_neighbors, batch=batch.batch
        )
        batch.edge_index = edge_index
        return batch

    def save(self, path: str | Path) -> None:
        """Save GNN weights + config to directory."""
        import torch

        if self._net is None:
            raise RuntimeError("Cannot save: model has not been fitted.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), save_dir / "weights.pt")

        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self._save_metadata(save_dir)
        log.info("GNN saved", path=str(save_dir))

    def load(self, path: str | Path) -> None:
        """Load GNN weights + config from directory."""
        import torch

        load_dir = Path(path)

        with open(load_dir / "config.json") as f:
            cfg_dict = json.load(f)

        # Restore config — must include ALL architecture fields to avoid state_dict mismatch
        self.config = GNNConfig(
            k_neighbors=cfg_dict.get("k_neighbors", self.config.k_neighbors),
            n_particles=cfg_dict.get("n_particles", self.config.n_particles),
            n_node_features=cfg_dict.get("n_node_features", self.config.n_node_features),
            n_edge_conv_layers=cfg_dict.get("n_edge_conv_layers", self.config.n_edge_conv_layers),
            hidden_dim=cfg_dict.get("hidden_dim", self.config.hidden_dim),
            mlp_head_dims=cfg_dict.get("mlp_head_dims", self.config.mlp_head_dims),
            dropout=cfg_dict.get("dropout", self.config.dropout),
            seed=cfg_dict.get("seed", self.config.seed),
        )

        self._net = EdgeConvNet.build(self.config)
        state_dict = torch.load(
            load_dir / "weights.pt",
            map_location="cpu",
            weights_only=True,
        )
        self._net.load_state_dict(state_dict)
        self._device = torch.device("cpu")

        meta = self._load_metadata(load_dir)
        self._metadata.update(meta)
        self._is_fitted = True

        log.info("GNN loaded", path=str(load_dir))

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        if self._net is not None:
            base["n_parameters"] = self._net.n_parameters()
        base["config"] = self.config.to_dict()
        return base
