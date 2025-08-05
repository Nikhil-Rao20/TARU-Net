# -*- coding: utf-8 -*-
"""
Main Driver Script for Medical Image Segmentation Experiments.

This script orchestrates the training and evaluation of segmentation models
on two different datasets:
1. Lung Cancer Segmentation (from .npy slices)
2. COVID-19 Infection Segmentation (from 3D NIfTI volumes)

It leverages the refactored, class-based training and evaluation modules.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import logging
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# --- Refactored Local Modules ---
# Note: Ensure these files are in your Python path.
from dataloader import (COVID2DDataset, LungCancerSegmentationDataset,
                        build_covid_image_mask_pairs)
from Models.unet_family_models import U_Net
from Models.unet_model import UNet1
from test_code import ModelEvaluator, TestConfig
from train_code import Trainer, TrainConfig

# =============================================================================
# 2. INITIAL CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)

# --- General Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count() // 2  # Use half of the available CPU cores

# --- Lung Cancer Experiment Configuration ---
LUNG_CANCER_DATA_DIR = '/kaggle/input/lung-cancer-segment'
LUNG_CANCER_EPOCHS = 30

# --- COVID-19 Experiment Configuration ---
COVID_DATA_DIR = "/kaggle/input/covid19-ct-scans"
COVID_EPOCHS = 30
COVID_TRAIN_TEST_SPLIT_RATIO = 0.2
COVID_VAL_TEST_SPLIT_RATIO = 0.5


# =============================================================================
# 3. EXPERIMENT RUNNER FUNCTIONS
# =============================================================================

def run_lung_cancer_experiment():
    """
    Configures and runs the full training and evaluation pipeline for the
    Lung Cancer Segmentation dataset.
    """
    logging.info("\n" + "="*80)
    logging.info("STARTING: Lung Cancer Segmentation Experiment")
    logging.info("="*80)

    # --- 1. Setup Datasets and DataLoaders ---
    logging.info("Setting up Lung Cancer datasets...")
    train_dataset = LungCancerSegmentationDataset(root_dir=os.path.join(LUNG_CANCER_DATA_DIR, 'train'))
    val_dataset = LungCancerSegmentationDataset(root_dir=os.path.join(LUNG_CANCER_DATA_DIR, 'val'))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    logging.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 2. Configure and Run Training ---
    model = UNet1(n_class=1)
    train_config = TrainConfig(
        experiment_name="LungCancer_UNet",
        num_epochs=LUNG_CANCER_EPOCHS,
        device=DEVICE,
        best_metric_to_track="val_dice"
    )
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train()

    # --- 3. Configure and Run Testing ---
    logging.info("Starting evaluation on the validation set...")
    test_config = TestConfig(
        experiment_name="LungCancer_UNet",
        device=DEVICE
    )
    # Load the best model saved by the trainer for evaluation
    best_model_path = os.path.join(test_config.output_dir, test_config.experiment_name, "best_model.pth")
    if not os.path.exists(best_model_path):
        logging.error(f"Best model not found at {best_model_path}. Skipping evaluation.")
        return

    eval_model = UNet1(n_class=1)
    eval_model.load_state_dict(torch.load(best_model_path))
    
    evaluator = ModelEvaluator(model=eval_model, config=test_config)
    evaluator.evaluate(test_loader=val_loader) # Evaluating on the validation set as per original script

    logging.info("COMPLETED: Lung Cancer Segmentation Experiment")


def run_covid_experiment():
    """
    Configures and runs the full training and evaluation pipeline for the
    COVID-19 Infection Segmentation dataset.
    """
    logging.info("\n" + "="*80)
    logging.info("STARTING: COVID-19 Infection Segmentation Experiment")
    logging.info("="*80)

    # --- 1. Setup Datasets and DataLoaders ---
    logging.info("Setting up COVID-19 datasets...")
    img_dir = os.path.join(COVID_DATA_DIR, "ct_scans")
    mask_dir = os.path.join(COVID_DATA_DIR, "infection_mask")

    all_pairs = build_covid_image_mask_pairs(img_dir, mask_dir)
    train_pairs, val_test_pairs = train_test_split(
        all_pairs, test_size=COVID_TRAIN_TEST_SPLIT_RATIO, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        val_test_pairs, test_size=COVID_VAL_TEST_SPLIT_RATIO, random_state=42
    )

    train_ds = COVID2DDataset(img_dir, mask_dir, train_pairs)
    val_ds = COVID2DDataset(img_dir, mask_dir, val_pairs)
    test_ds = COVID2DDataset(img_dir, mask_dir, test_pairs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    logging.info(f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # --- 2. Configure and Run Training ---
    model = U_Net()
    train_config = TrainConfig(
        experiment_name="COVID_UNet",
        num_epochs=COVID_EPOCHS,
        device=DEVICE,
        best_metric_to_track="val_dice"
    )
    trainer = Trainer(model, train_loader, val_loader, train_config)
    trainer.train()

    # --- 3. Configure and Run Testing ---
    logging.info("Starting evaluation on the test set...")
    test_config = TestConfig(
        experiment_name="COVID_UNet",
        device=DEVICE
    )
    best_model_path = os.path.join(test_config.output_dir, test_config.experiment_name, "best_model.pth")
    if not os.path.exists(best_model_path):
        logging.error(f"Best model not found at {best_model_path}. Skipping evaluation.")
        return

    eval_model = U_Net()
    eval_model.load_state_dict(torch.load(best_model_path))

    evaluator = ModelEvaluator(model=eval_model, config=test_config)
    evaluator.evaluate(test_loader=test_loader)

    logging.info("COMPLETED: COVID-19 Infection Segmentation Experiment")


# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to run all experiments."""
    run_lung_cancer_experiment()
    run_covid_experiment()


if __name__ == "__main__":
    main()