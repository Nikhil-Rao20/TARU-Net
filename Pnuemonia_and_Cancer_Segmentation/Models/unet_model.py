import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)    # For Segmentation mask
        self.finalconv = nn.Conv2d(64, 1, kernel_size=1)   # For Denoised image

    def forward(self, x):
        x = F.relu(self.e11(x))
        x1 = F.relu(self.e12(x))
        x = self.pool1(x1)

        x = F.relu(self.e21(x))
        x2 = F.relu(self.e22(x))
        x = self.pool2(x2)

        x = F.relu(self.e31(x))
        x3 = F.relu(self.e32(x))
        x = self.pool3(x3)

        x = F.relu(self.e41(x))
        x4 = F.relu(self.e42(x))
        x = self.pool4(x4)

        x = F.relu(self.e51(x))
        x = F.relu(self.e52(x))

        x = self.upconv1(x)
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.d11(x))
        x = F.relu(self.d12(x))

        x = self.upconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.d21(x))
        x = F.relu(self.d22(x))

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.d31(x))
        x = F.relu(self.d32(x))

        x = self.upconv4(x)
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.d41(x))
        x = F.relu(self.d42(x))

        out = self.outconv(x) 
        return out
