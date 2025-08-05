# -*- coding: utf-8 -*-
"""
Reusable PyTorch Training Pipeline for Semantic Segmentation.

This module provides a robust, configurable Trainer class for training and
validating binary segmentation models. It handles metric tracking, logging,
and model checkpointing.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Note: Assumes a local 'metrics.py' file with these functions.
from metrics import binary_dice, binary_f1, binary_iou, recall, specificity, precision

# =============================================================================
# 2. INITIAL CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# =============================================================================
# 3. CONFIGURATION AND METRIC HANDLING
# =============================================================================

@dataclass
class TrainConfig:
    """Configuration settings for the model training process."""
    experiment_name: str = "default_experiment"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "training_results"
    num_epochs: int = 50
    learning_rate: float = 1e-4
    save_best_model: bool = True
    best_metric_to_track: str = "val_loss"  # or 'val_dice'


class EpochMetrics:
    """A helper class to track and aggregate metrics for one epoch."""
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def update(self, new_metrics: Dict[str, float]):
        for key, value in new_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_averages(self) -> Dict[str, float]:
        return {key: np.nanmean(val) for key, val in self.metrics.items() if val}


# =============================================================================
# 4. MODEL TRAINER CLASS
# =============================================================================

class Trainer:
    """A class to handle the training and validation of a segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
    ):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            config (TrainConfig): Configuration object for the training run.
        """
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        self.experiment_path = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.experiment_path, exist_ok=True)
        logging.info(f"Trainer initialized for experiment: '{config.experiment_name}'")

    def train(self) -> pd.DataFrame:
        """
        Runs the full training and validation loop for the specified number of epochs.

        Returns:
            pd.DataFrame: A DataFrame containing the training history.
        """
        history = []
        best_metric_value = float('inf') if "loss" in self.config.best_metric_to_track else -float('inf')

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            # --- Training Phase ---
            train_metrics = self._run_one_epoch(phase="train", epoch_num=epoch)
            # --- Validation Phase ---
            val_metrics = self._run_one_epoch(phase="val", epoch_num=epoch)

            epoch_duration = time.time() - start_time

            # --- Logging and Checkpointing ---
            epoch_summary = {**train_metrics, **val_metrics, "time_s": epoch_duration}
            history.append(epoch_summary)
            self._log_epoch_summary(epoch + 1, epoch_summary)

            current_metric = epoch_summary.get(self.config.best_metric_to_track)
            if self.config.save_best_model and self._is_better(current_metric, best_metric_value):
                best_metric_value = current_metric
                self._save_checkpoint("best_model.pth")
                logging.info(f"✅ New best model saved with {self.config.best_metric_to_track}: {best_metric_value:.4f}")

        self._save_checkpoint("final_model.pth")
        history_df = pd.DataFrame(history)
        self._save_history(history_df)
        return history_df

    def _run_one_epoch(self, phase: str, epoch_num: int) -> Dict[str, float]:
        """Runs a single epoch of training or validation."""
        is_train = phase == "train"
        self.model.train(is_train)
        dataloader = self.train_loader if is_train else self.val_loader
        epoch_metrics = EpochMetrics()

        pbar_desc = f"Epoch {epoch_num + 1}/{self.config.num_epochs} [{phase.capitalize()}]"
        with torch.set_grad_enabled(is_train):
            for images, masks in tqdm(dataloader, desc=pbar_desc, leave=False):
                batch_metrics = self._run_one_batch(images, masks, is_train)
                epoch_metrics.update(batch_metrics)

        # Prefix metrics with 'train_' or 'val_'
        avg_metrics = epoch_metrics.get_averages()
        return {f"{phase}_{key}": val for key, val in avg_metrics.items()}

    def _run_one_batch(
        self, images: torch.Tensor, masks: torch.Tensor, is_train: bool
    ) -> Dict[str, float]:
        """Processes a single batch of data."""
        images = images.to(self.config.device)
        masks = masks.to(self.config.device).float()

        if is_train:
            self.optimizer.zero_grad()

        logits = self.model(images)
        loss = self.criterion(logits, masks)

        if is_train:
            loss.backward()
            self.optimizer.step()

        # Calculate metrics
        preds = (torch.sigmoid(logits) > 0.5).bool()
        return {
            "loss": loss.item(),
            "dice": binary_dice(preds, masks.bool()).item(),
            "iou": binary_iou(preds, masks.bool()).item(),
            "f1": binary_f1(preds, masks.bool()).item(),
            "precision": precision(preds, masks).item(),
            "recall": recall(preds, masks).item(),
            "specificity": specificity(preds, masks).item()
        }
    
    def _is_better(self, current: float, best: float) -> bool:
        """Checks if the current metric is better than the best one so far."""
        if "loss" in self.config.best_metric_to_track:
            return current < best  # For loss, lower is better
        else:
            return current > best  # For scores like Dice/IoU, higher is better

    def _save_checkpoint(self, filename: str):
        """Saves the model state dictionary."""
        path = os.path.join(self.experiment_path, filename)
        torch.save(self.model.state_dict(), path)

    def _save_history(self, history_df: pd.DataFrame):
        """Saves the training history to a CSV file."""
        csv_path = os.path.join(self.experiment_path, "training_log.csv")
        history_df.to_csv(csv_path, index_label="epoch")
        logging.info(f"Full training history saved to: {csv_path}")

    @staticmethod
    def _log_epoch_summary(epoch_num: int, summary: Dict[str, float]):
        """Prints a formatted summary of the epoch's results."""
        log_str = (
            f"Epoch {epoch_num:02d} | "
            f"Train Loss: {summary.get('train_loss', 0):.4f}, "
            f"Train Dice: {summary.get('train_dice', 0):.4f} | "
            f"Val Loss: {summary.get('val_loss', 0):.4f}, "
            f"Val Dice: {summary.get('val_dice', 0):.4f} | "
            f"Time: {summary.get('time_s', 0):.2f}s"
        )
        logging.info(log_str)
