import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    # Test data transformations
    test_image_transform = transforms.Compose(
            [   transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    train_image_transform_0 = transforms.Compose(
            [   transforms.Resize((224, 224)),
                # AddGaussianNoise(0., 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    train_data_org = datasets.ImageFolder(train_dir, transform=train_image_transform_0)
    test_data = datasets.ImageFolder(test_dir, transform=test_image_transform)
    train_data  =train_data_org
    class_names = train_data_org.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
    )
    test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
    )

    return train_dataloader,test_dataloader,class_names

