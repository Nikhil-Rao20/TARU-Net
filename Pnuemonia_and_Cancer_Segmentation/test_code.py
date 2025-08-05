# -*- coding: utf-8 -*-
"""
Model Evaluation Pipeline for Semantic Segmentation.

This module provides a robust class for testing and evaluating binary
segmentation models using PyTorch. It handles metric calculation, logging,
and saving predicted outputs.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Note: Assumes a local 'metrics.py' file with these functions.
# Ensure this file is in your Python path.
from metrics import (binary_dice, binary_f1, binary_iou, dice_score, f1_score,
                     precision, recall, specificity)

# =============================================================================
# 2. INITIAL CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# =============================================================================
# 3. CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class TestConfig:
    """
    Configuration settings for the model evaluation process.

    Attributes:
        experiment_name (str): A unique name for the test run.
        device (str): The device to run the model on (e.g., 'cuda', 'cpu').
        output_dir (str): Root directory to save results and predictions.
        save_predictions (bool): Whether to save visual predictions.
        metrics_to_compute (List[str]): List of metrics to calculate.
    """
    experiment_name: str = "default_test"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "test_results"
    save_predictions: bool = True
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        "loss", "dice", "iou", "f1", "precision", "recall", "specificity"
    ])


# =============================================================================
# 4. MODEL EVALUATOR CLASS
# =============================================================================

class ModelEvaluator:
    """
    A class to handle the evaluation of a binary segmentation model.
    """
    def __init__(self, model: nn.Module, config: TestConfig):
        """
        Args:
            model (nn.Module): The PyTorch model to evaluate.
            config (TestConfig): The configuration object for the test run.
        """
        self.model = model.to(config.device)
        self.config = config
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Create dedicated output directory for this experiment
        self.experiment_path = os.path.join(config.output_dir, config.experiment_name)
        self.preds_path = os.path.join(self.experiment_path, "predictions")
        os.makedirs(self.preds_path, exist_ok=True)

        logging.info(f"Evaluator initialized for experiment: '{config.experiment_name}'")
        logging.info(f"Results will be saved to: {self.experiment_path}")

    def evaluate(self, test_loader: DataLoader) -> pd.DataFrame:
        """
        Runs the full evaluation loop on a given test dataset.

        Args:
            test_loader (DataLoader): The DataLoader for the test set.

        Returns:
            pd.DataFrame: A DataFrame containing the aggregated results.
        """
        self.model.eval()
        batch_metrics: Dict[str, List[float]] = {
            metric: [] for metric in self.config.metrics_to_compute
        }
        total_time = 0.0

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(test_loader, desc=f"Evaluating '{self.config.experiment_name}'")):
                start_time = time.time()

                images = images.to(self.config.device)
                masks = masks.to(self.config.device).float()

                # Forward pass
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).bool()

                # Calculate metrics for the batch
                self._compute_and_store_batch_metrics(
                    batch_metrics, logits, masks, preds
                )

                total_time += time.time() - start_time

                if self.config.save_predictions:
                    self._save_batch_predictions(images, masks, preds, batch_idx=i)

        # Aggregate and save results
        aggregated_results = self._aggregate_and_log_results(batch_metrics, total_time)
        return aggregated_results

    def _compute_and_store_batch_metrics(
        self,
        batch_metrics: Dict[str, List],
        logits: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor
    ):
        """Calculates and stores metrics for a single batch."""
        # Loss
        if "loss" in batch_metrics:
            loss = self.loss_fn(logits, masks)
            batch_metrics["loss"].append(loss.item())

        # Standard metrics
        if "dice" in batch_metrics:
            batch_metrics["dice"].append(binary_dice(preds, masks.bool()).item())
        if "iou" in batch_metrics:
            batch_metrics["iou"].append(binary_iou(preds, masks.bool()).item())
        if "f1" in batch_metrics:
            batch_metrics["f1"].append(binary_f1(preds, masks.bool()).item())

        # Additional metrics (require probabilities)
        if "precision" in batch_metrics or "recall" in batch_metrics:
            prec = precision(preds, masks).item()
            rec = recall(preds, masks).item()
            if "precision" in batch_metrics:
                batch_metrics["precision"].append(prec)
            if "recall" in batch_metrics:
                batch_metrics["recall"].append(rec)
        if "specificity" in batch_metrics:
            batch_metrics["specificity"].append(specificity(preds, masks).item())

    def _save_batch_predictions(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        preds: torch.Tensor,
        batch_idx: int
    ):
        """Saves image, ground truth, and prediction triplets for a batch."""
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy().astype(np.uint8)

        for i in range(images.size(0)):
            img = (images_np[i, 0] * 255).astype(np.uint8)
            mask = (masks_np[i, 0] * 255).astype(np.uint8)
            pred = preds_np[i, 0] * 255

            cv2.imwrite(os.path.join(self.preds_path, f"batch{batch_idx}_img{i}.png"), img)
            cv2.imwrite(os.path.join(self.preds_path, f"batch{batch_idx}_gt{i}.png"), mask)
            cv2.imwrite(os.path.join(self.preds_path, f"batch{batch_idx}_pred{i}.png"), pred)

    def _aggregate_and_log_results(
        self,
        batch_metrics: Dict[str, List[float]],
        total_time: float
    ) -> pd.DataFrame:
        """Aggregates metrics, logs them, and saves to a CSV file."""
        # Calculate mean of all collected metrics, handling potential empty lists
        final_metrics = {
            key: np.nanmean(val) for key, val in batch_metrics.items() if val
        }
        final_metrics["Total Time (s)"] = total_time
        final_metrics["Avg Time/Batch (s)"] = total_time / len(next(iter(batch_metrics.values()), []))

        # Log results to console
        logging.info("\n" + "="*30 + " Test Results " + "="*30)
        for key, value in final_metrics.items():
            logging.info(f"{key:<20}: {value:.4f}")
        logging.info("="*74)

        # Save to CSV
        results_df = pd.DataFrame([final_metrics])
        csv_path = os.path.join(self.experiment_path, "test_summary.csv")
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        logging.info(f"Results summary saved to: {csv_path}")

        return results_df
