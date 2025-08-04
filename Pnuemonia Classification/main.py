##########################Importing Libraries##########################
# This code is used to merge multiple datasets of Pneumonia and train models on it.
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchinfo import summary
from typing import Dict, List, Tuple
import requests
from tqdm.auto import tqdm
import torchvision
import time
import pandas as pd
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import torchvision.transforms as tt
from train_code import train
from Models.mobilenetv2_model import MobileNetV2
from Models.resnet_model import ResNet101, ResNet50
from Models.shufflenet_model import ShuffleNet
from dataloader import create_dataloaders
from dataset_creator import combine_datasets

###############################Setting Device##########################
# Setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################Combining Datasets##########################
# Combining datasets from different sources into a single dataset
# The below given paths are from the kaggle dataset, it would work if you are in the kaggle notebooks. 
# # If you are running this code local, please change the paths accordingly. The respective datasets links are given below.
dataset_dirs_input = [
    "/kaggle/input/chest-xray-covid19-pneumonia/Data", #https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia
    "/kaggle/input/chest-x-ray-image/Data", #https://www.kaggle.com/datasets/alsaniipe/chest-x-ray-image
    "/kaggle/input/pediatric-pneumonia-chest-xray/Pediatric Chest X-ray Pneumonia", #https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray
    "/kaggle/input/labeled-chest-xray-images/chest_xray", #https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
    '/kaggle/input/chest-x-ray-pneumonia/chest_xray' #https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
]
dataset_dir_1 = "/kaggle/input/pneumonia-xray-images" #https://www.kaggle.com/datasets/pcbreviglieri/pneumonia-xray-images
dataset_dir_2 = "/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database" #https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
OUTPUT_DIR = "/kaggle/working/Combined_Dataset" # Output directory where the combined dataset will be saved

start_time = time.time()
combine_datasets(dataset_dirs=dataset_dirs_input,ds1_root = dataset_dir_1,ds2_root = dataset_dir_2, output_dir = OUTPUT_DIR)
print("Done! All datasets have been merged into", "Combined_Dataset")
print("Time required is: ", time.time()-start_time)

###############################Dataloader Code##########################
# After combining the datasets, we can now train models on the combined dataset.
# Please set the train and test directories accordingly.
train_dir = '/kaggle/working/Combined_Dataset/train'
test_dir = '/kaggle/working/Combined_Dataset/test'
batch_size = 32

# Creating dataloaders for training and testing
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,test_dir=test_dir, batch_size=batch_size) 


#######################Training ResNet50 Model##########################
# Training ResNet50 model on the combined dataset
model_1 = ResNet50(num_classes=3)
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.0001)
loss_function = torch.nn.CrossEntropyLoss()

results_resnet = train(model=model_1, 
                train_dataloader=train_dataloader_pretrained, 
                test_dataloader=test_dataloader_pretrained, 
                optimizer=optimizer, 
                loss_fn=loss_function, 
                epochs=10, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
pd.DataFrame(results_resnet).to_csv("Puemonia_ResNet_Results.csv")
torch.save(model_1.state_dict(), "Puemonia_ResNet_Model.pth")


########################Training ResNet101 Model##########################
# Training ResNet101 model on the combined dataset
model_1 = ResNet101(num_classes=3)
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.0001)
loss_function = torch.nn.CrossEntropyLoss()

results_resnet = train(model=model_1, 
                train_dataloader=train_dataloader_pretrained, 
                test_dataloader=test_dataloader_pretrained, 
                optimizer=optimizer, 
                loss_fn=loss_function, 
                epochs=10, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
pd.DataFrame(results_resnet).to_csv("Puemonia_ResNet101_Results.csv")
torch.save(model_1.state_dict(), "Puemonia_ResNet101_Model.pth")

#########################Training MobileNetV2 Model##########################
# Training MobileNetV2 model on the combined dataset
mobile_model = MobileNetV2()
optimizer = torch.optim.Adam(mobile_model.parameters(), lr=0.0001)
loss_function = torch.nn.CrossEntropyLoss()

results_resnet = train(model=mobile_model, 
                train_dataloader=train_dataloader_pretrained, 
                test_dataloader=test_dataloader_pretrained, 
                optimizer=optimizer, 
                loss_fn=loss_function, 
                epochs=10, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

pd.DataFrame(results_resnet).to_csv("Puemonia_MobileNet_Results.csv")
torch.save(mobile_model.state_dict(), "Puemonia_MobileNet_Model.pth")

##########################Training ShuffleNet Model##########################   
# Training ShuffleNet model on the combined dataset
mobile_model = ShuffleNet()
optimizer = torch.optim.Adam(mobile_model.parameters(), lr=0.0001)
loss_function = torch.nn.CrossEntropyLoss()

results_resnet = train(model=mobile_model, 
                train_dataloader=train_dataloader_pretrained, 
                test_dataloader=test_dataloader_pretrained, 
                optimizer=optimizer, 
                loss_fn=loss_function, 
                epochs=10, 
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

pd.DataFrame(results_resnet).to_csv("Puemonia_ShuffleNet_Results.csv")
torch.save(mobile_model.state_dict(), "Puemonia_ShuffleNet_Model.pth")