# -*- coding: utf-8 -*-
"""
PyTorch Dataset classes for loading 2D medical imaging data.

This module contains two main classes:
1.  LungCancerSegmentationDataset: Loads pre-processed 2D slices from .npy files.
2.  COVID2DDataset: Loads, caches, and serves 2D slices from 3D NIfTI volumes.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import logging
import os
from typing import List, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 2. INITIAL CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# 3. LUNG CANCER DATASET (from .npy files)
# =============================================================================

class LungCancerSegmentationDataset(Dataset):
    """
    Dataset for loading 2D lung cancer slices stored as individual .npy files.

    Assumes a directory structure like:
    /root_dir
    ├── /patient_01
    │   ├── /data
    │   │   ├── slice_001.npy
    │   │   └── ...
    │   └── /masks
    │       ├── slice_001.npy
    │       └── ...
    └── /patient_02
        └── ...
    """
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir (str): The root directory containing patient folders.
        """
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        patient_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        logging.info(f"Found {len(patient_dirs)} patients in {root_dir}.")

        for patient in patient_dirs:
            data_dir = os.path.join(root_dir, patient, 'data')
            mask_dir = os.path.join(root_dir, patient, 'masks')

            if not os.path.isdir(data_dir) or not os.path.isdir(mask_dir):
                logging.warning(f"Skipping patient {patient}: missing 'data' or 'masks' directory.")
                continue

            for file_name in sorted(os.listdir(data_dir)):
                if file_name.endswith('.npy'):
                    img_path = os.path.join(data_dir, file_name)
                    mask_path = os.path.join(mask_dir, file_name)
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
                    else:
                        logging.warning(f"Mask not found for image: {img_path}")
        
        logging.info(f"Initialized dataset with {len(self.image_paths)} total slices.")

    def __len__(self) -> int:
        """Returns the total number of slices in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and returns a single image-mask pair as tensors.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image
            tensor and the mask tensor, both with shape [1, H, W].
        """
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        # Add channel dimension and convert to float tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor


# =============================================================================
# 4. COVID-19 PNEUMONIA DATASET (from .nii files)
# =============================================================================

def build_covid_image_mask_pairs(image_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
    """
    Matches image files to mask files based on custom naming conventions.

    Args:
        image_dir (str): Directory containing NIfTI image files.
        mask_dir (str): Directory containing NIfTI mask files.

    Returns:
        List[Tuple[str, str]]: A list of (image_filename, mask_filename) tuples.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii")])
    mask_files = set(os.listdir(mask_dir)) # Use a set for fast lookups

    def _normalize_scan_name(name: str) -> str:
        """Converts complex image filenames to their simpler mask filename equivalent."""
        # Handles names like 'coronacases_org_001.nii' -> 'coronacases_001'
        if name.startswith("coronacases_org_"):
            return name.replace("coronacases_org_", "coronacases_").replace(".nii", "")
        # Handles names like 'radiopaedia_org_covid-19-pneumonia-10-dcm.nii' -> 'radiopaedia_10'
        if name.startswith("radiopaedia_org_"):
            base_name = name.replace("radiopaedia_org_covid-19-pneumonia-", "").replace("-dcm", "")
            return "radiopaedia_" + base_name.replace(".nii", "")
        return name.replace(".nii", "")

    image_mask_pairs = []
    for img_file in image_files:
        norm_name = _normalize_scan_name(img_file)
        expected_mask_file = norm_name + ".nii"
        if expected_mask_file in mask_files:
            image_mask_pairs.append((img_file, expected_mask_file))
    return image_mask_pairs


class COVID2DDataset(Dataset):
    """
    Dataset for COVID-19 segmentation that loads 2D slices from 3D NIfTI files.

    This class pre-loads and caches all 3D volumes into memory to ensure
    fast data access during training, avoiding repeated disk I/O.
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        target_size: Tuple[int, int] = (256, 256)
    ):
        """
        Args:
            image_dir (str): Path to the directory with NIfTI image volumes.
            mask_dir (str): Path to the directory with NIfTI mask volumes.
            target_size (Tuple[int, int]): The target (height, width) to resize slices to.
        """
        self.target_size = target_size
        self.image_mask_pairs = build_covid_image_mask_pairs(image_dir, mask_dir)
        
        self.cached_volumes = {}
        self.slice_index: List[Tuple[str, str, int]] = []

        logging.info("Caching NIfTI volumes into memory. This may take a moment...")
        for img_file, mask_file in self.image_mask_pairs:
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            
            # Load and cache the full 3D data arrays
            self.cached_volumes[img_file] = nib.load(img_path).get_fdata()
            self.cached_volumes[mask_file] = nib.load(mask_path).get_fdata()

            num_slices = self.cached_volumes[img_file].shape[2]
            for i in range(num_slices):
                self.slice_index.append((img_file, mask_file, i))
        
        logging.info(f"Finished caching. Total 2D slices available: {len(self.slice_index)}")

    def __len__(self) -> int:
        """Returns the total number of 2D slices across all volumes."""
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a preprocessed 2D slice from the cached 3D volumes.

        Args:
            idx (int): The index of the slice to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image
            tensor and the mask tensor, both with shape [1, H, W].
        """
        img_key, mask_key, slice_idx = self.slice_index[idx]

        # Retrieve data from cache (fast) instead of disk (slow)
        img_vol = self.cached_volumes[img_key]
        mask_vol = self.cached_volumes[mask_key]

        img_slice = img_vol[:, :, slice_idx]
        mask_slice = mask_vol[:, :, slice_idx]

        # Resize image using linear interpolation
        img_resized = cv2.resize(img_slice, self.target_size, interpolation=cv2.INTER_AREA)
        # Normalize image to [0, 1]
        img_min, img_max = np.min(img_resized), np.max(img_resized)
        img_resized = (img_resized - img_min) / (img_max - img_min + 1e-8)

        # Resize mask using nearest-neighbor to preserve binary values
        mask_resized = cv2.resize(mask_slice, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.float32)

        # Add channel dimension and convert to float tensors
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return img_tensor, mask_tensor


# =============================================================================
# 5. DEMONSTRATION BLOCK
# =============================================================================

if __name__ == '__main__':
    # This block allows for direct testing of the Dataset classes.
    
    def test_lung_cancer_dataset():
        logging.info("\n" + "="*50)
        logging.info("--- Testing LungCancerSegmentationDataset ---")
        # Create dummy data
        dummy_root = "dummy_lung_cancer_data"
        os.makedirs(os.path.join(dummy_root, "patient_1", "data"), exist_ok=True)
        os.makedirs(os.path.join(dummy_root, "patient_1", "masks"), exist_ok=True)
        
        dummy_image = np.random.rand(128, 128)
        dummy_mask = (np.random.rand(128, 128) > 0.5).astype(float)
        
        np.save(os.path.join(dummy_root, "patient_1", "data", "slice_01.npy"), dummy_image)
        np.save(os.path.join(dummy_root, "patient_1", "masks", "slice_01.npy"), dummy_mask)
        
        try:
            dataset = LungCancerSegmentationDataset(root_dir=dummy_root)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            img, mask = next(iter(dataloader))
            
            logging.info(f"Successfully loaded one batch.")
            logging.info(f"Image batch shape: {img.shape}, dtype: {img.dtype}") # Expected: [1, 1, 128, 128]
            logging.info(f"Mask batch shape: {mask.shape}, dtype: {mask.dtype}")   # Expected: [1, 1, 128, 128]
            assert img.shape == (1, 1, 128, 128) and mask.shape == (1, 1, 128, 128)
        except Exception as e:
            logging.error(f"Test failed: {e}")

    def test_covid_dataset():
        logging.info("\n" + "="*50)
        logging.info("--- Testing COVID2DDataset ---")
        # Create dummy NIfTI data
        dummy_root = "dummy_covid_data"
        img_dir = os.path.join(dummy_root, "images")
        mask_dir = os.path.join(dummy_root, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        # Create a 3D volume (e.g., 64x64 with 5 slices)
        dummy_volume = np.random.rand(64, 64, 5)
        dummy_mask_vol = (np.random.rand(64, 64, 5) > 0.5).astype(np.uint8)
        
        affine = np.eye(4) # Dummy affine matrix
        
        img_nii = nib.Nifti1Image(dummy_volume, affine)
        mask_nii = nib.Nifti1Image(dummy_mask_vol, affine)
        
        nib.save(img_nii, os.path.join(img_dir, "coronacases_org_001.nii"))
        nib.save(mask_nii, os.path.join(mask_dir, "coronacases_001.nii"))
        
        try:
            dataset = COVID2DDataset(image_dir=img_dir, mask_dir=mask_dir, target_size=(256, 256))
            assert len(dataset) == 5 # 5 slices
            
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            img, mask = next(iter(dataloader))

            logging.info(f"Successfully loaded one batch.")
            logging.info(f"Image batch shape: {img.shape}, dtype: {img.dtype}") # Expected: [2, 1, 256, 256]
            logging.info(f"Mask batch shape: {mask.shape}, dtype: {mask.dtype}")   # Expected: [2, 1, 256, 256]
            assert img.shape == (2, 1, 256, 256) and mask.shape == (2, 1, 256, 256)
        except Exception as e:
            logging.error(f"Test failed: {e}")

    # Run tests
    test_lung_cancer_dataset()
    test_covid_dataset()