################################### Lung Cancer Segmentation Dataset Loader ###################################
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image

class LungCancerSegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.mask_paths = []
        patient_dirs = sorted(os.listdir(root_dir))
        for patient in patient_dirs:
            data_dir = os.path.join(root_dir, patient, 'data')
            mask_dir = os.path.join(root_dir, patient, 'masks')
            
            for file_name in sorted(os.listdir(data_dir)):
                img_path = os.path.join(data_dir, file_name)
                mask_path = os.path.join(mask_dir, file_name)  # same name
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask
    
######################################COVID Pneumonia Dataset######################################


import os
import torch
import nibabel as nib
import numpy as np
import cv2
from torch.utils.data import Dataset

def build_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii")])

    # Normalize image names to match with masks
    def normalize(name):
        if name.startswith("coronacases_org_"):
            return name.replace("coronacases_org_", "coronacases_").replace(".nii", "")
        if name.startswith("radiopaedia_org_covid-19-pneumonia-"):
            name = name.replace("radiopaedia_org_covid-19-pneumonia-", "")
            name = name.replace("-dcm", "")
            return "radiopaedia_" + name.replace(".nii", "")
        return name.replace(".nii", "")

    image_mask_pairs = []
    for img in image_files:
        norm_name = normalize(img)
        mask_filename = norm_name + ".nii"
        if mask_filename in mask_files:
            image_mask_pairs.append((img, mask_filename))
    return image_mask_pairs

class COVID2DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_mask_pairs, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_mask_pairs = image_mask_pairs
        self.transform = transform
        self.target_size = (256, 256)
        self.slices = []

        # Pre-index all slice references (image, mask, slice index)
        for img_file, mask_file in self.image_mask_pairs:
            img_path = os.path.join(image_dir, img_file)
            img_nii = nib.load(img_path)
            num_slices = img_nii.shape[2]
            for idx in range(num_slices):
                self.slices.append((img_file, mask_file, idx))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_file, mask_file, slice_idx = self.slices[idx]
        img = nib.load(os.path.join(self.image_dir, img_file)).get_fdata()
        mask = nib.load(os.path.join(self.mask_dir, mask_file)).get_fdata()

        img_slice = img[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]

        # Resize and normalize
        img_resized = cv2.resize(img_slice, self.target_size, interpolation=cv2.INTER_AREA)
        img_resized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized) + 1e-8)

        mask_resized = cv2.resize(mask_slice, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8)

        # Add channel dim
        img_tensor = torch.FloatTensor(img_resized).unsqueeze(0)  # [1, H, W]
        mask_tensor = torch.LongTensor(mask_resized).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor