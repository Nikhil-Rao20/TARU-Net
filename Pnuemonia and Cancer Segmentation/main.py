from warnings import filterwarnings
filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import gc
import random
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.transforms.v2 import GaussianNoise
from torchmetrics import JaccardIndex, Precision, Recall, Specificity, F1Score, AUROC
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from skimage.metrics import mean_squared_error as mse_skimage
from skimage.metrics import hausdorff_distance
from scipy.ndimage import distance_transform_edt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataloader import LungCancerSegmentationDataset, COVID2DDataset, build_image_mask_pairs
from Models.unet_model import UNet1
from Models.unet_family_models import U_Net
from train_code import cancer_seg_train_model, covid_train_binary_segmentation
from test_code import seg_test_model, covid_test_binary_model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################Lung Cancer Segmentation Dataset Loader########################
CANCER_TRAIN_DIR = '/kaggle/input/lung-cancer-segment/train'
CANCER_VAL_DIR = '/kaggle/input/lung-cancer-segment/val'
train_dataset = LungCancerSegmentationDataset(root_dir=CANCER_TRAIN_DIR)
val_dataset = LungCancerSegmentationDataset(root_dir=CANCER_VAL_DIR)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
#########################Lung Cancer Segmentation UNet Model Training and Testing########################
model = UNet1(n_class=1).to(device)
NUM_EPOCHS = 30
model = model.to(device)
trained_model = cancer_seg_train_model(model,train_loader=train_loader,val_loader=val_loader,
                            device=device, name = 'UNET', num_epochs=NUM_EPOCHS)
test_results = seg_test_model(trained_model, val_loader, device, name='UNET')



########################COVID Pneumonia Dataset Loader########################

COVID_IMG_DIR = "/kaggle/input/covid19-ct-scans/ct_scans"
COVID_MASK_DIR = "/kaggle/input/covid19-ct-scans/infection_mask"

pairs = build_image_mask_pairs(COVID_IMG_DIR, COVID_MASK_DIR)
from sklearn.model_selection import train_test_split
train_pairs, val_test = train_test_split(pairs, test_size=0.2, random_state=42)
val_pairs, test_pairs = train_test_split(val_test, test_size=0.5, random_state=42)
train_ds = COVID2DDataset(COVID_IMG_DIR, COVID_MASK_DIR, train_pairs)
val_ds   = COVID2DDataset(COVID_IMG_DIR, COVID_MASK_DIR, val_pairs)
test_ds = COVID2DDataset(COVID_IMG_DIR, COVID_MASK_DIR, test_pairs)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)


model = U_Net()
covid_train_binary_segmentation(model, train_loader, val_loader, num_epochs=30, device=torch.device('cuda'), save_path='')
covid_test_binary_model(model, test_loader, device='cuda', save_preds=True, save_dir="preds")