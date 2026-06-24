import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from src.models.normalizing_flow.config import NormalizingFlowConfig
from src.utils.logging_config import get_logger

log = get_logger(__name__)

class NFTrainer:
    def __init__(self, config: NormalizingFlowConfig):
        self.config = config

    def train(self, model_wrapper, X_train_bg: np.ndarray, X_val_bg: np.ndarray, X_val_full: np.ndarray, y_val_full: np.ndarray, **kwargs) -> dict[str, Any]:
        """
        Train the Normalizing Flow on background data.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_wrapper._device = device
        
        input_dim = X_train_bg.shape[1]
        net = model_wrapper._build(input_dim).to(device)
        model_wrapper._net = net
        
        optimizer = optim.AdamW(
            net.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        if self.config.scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        else:
            scheduler = None
            
        train_tensor = torch.tensor(X_train_bg, dtype=torch.float32)
        val_tensor = torch.tensor(X_val_bg, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=self.config.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_epoch = 0
        min_log_prob = float('inf')
        max_log_prob = float('-inf')
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            net.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                
                log_prob = net.log_prob(x)
                loss = -log_prob.mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item() * x.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            if scheduler:
                scheduler.step()
                
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    log_prob = net.log_prob(x)
                    loss = -log_prob.mean()
                    val_loss += loss.item() * x.size(0)
                    
                    min_log_prob = min(min_log_prob, log_prob.min().item())
                    max_log_prob = max(max_log_prob, log_prob.max().item())
                    
            val_loss /= len(val_loader.dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
            if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                log.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
                
        fit_time_s = time.time() - start_time
        log.info(f"Training completed in {fit_time_s:.2f}s. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
        
        return {
            "fit_time_s": fit_time_s,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "min_log_prob": min_log_prob,
            "max_log_prob": max_log_prob,
        }
