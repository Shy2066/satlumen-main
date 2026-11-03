"""Supervised training for the directional-count SatLumen model."""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from src.utils.common import compute_metrics
from src.utils.metrics import adjust_learning_rate


class SatLumenTrainer:
    def __init__(self, model, optimizer, criterion, device, logger=None,
                 train_cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.train_cfg = train_cfg

    def _unpack(self, batch):
        inputs, labels = batch
        counts, count_idx = inputs
        return (counts.to(self.device), count_idx.to(self.device),
                labels.to(self.device).long())

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
        for batch_index, batch in enumerate(progress):
            if self.train_cfg:
                adjust_learning_rate(
                    self.optimizer, batch_index / len(loader) + epoch,
                    self.train_cfg)
            counts, count_idx, labels = self._unpack(batch)
            self.optimizer.zero_grad()
            logits = self.model(counts, count_idx)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(sample_count, 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        predictions, labels_all = [], []
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            counts, count_idx, labels = self._unpack(batch)
            logits = self.model(counts, count_idx)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
        return compute_metrics(np.concatenate(labels_all), np.concatenate(predictions))

    def train(self, train_loader, val_loader, epochs, save_dir=None):
        best_metric = float("-inf")
        best_epoch = None
        save_dir = save_dir or "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = os.path.join(save_dir, "best_model.pth")
        for epoch in range(epochs):
            if self.logger:
                self.logger.log_epoch_start(epoch, epochs)
            train_loss = self.train_epoch(train_loader, epoch)
            lr = self.optimizer.param_groups[0]["lr"]
            if self.logger:
                self.logger.log_train_loss(epoch, train_loss, lr)
            metrics = self.evaluate(val_loader)
            if self.logger:
                self.logger.log_validation(epoch, metrics)
            if metrics["F1-score"] > best_metric:
                best_metric = metrics["F1-score"]
                best_epoch = epoch
                torch.save(self.model.state_dict(), checkpoint)
                if self.logger:
                    self.logger.log_best_model(epoch, "F1-score", best_metric)
        if self.logger:
            self.logger.log_summary()
        self.best_epoch = best_epoch
        return best_metric
