""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PHConv2d

class DoubleConv(nn.Module):
    """(PHConv2d => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, ph_n, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        n = ph_n
        mid_channels = (mid_channels // n) * n
        out_channels = (out_channels // n) * n

        self.double_conv = nn.Sequential(
            PHConv2d(n=n, in_features=in_channels, out_features=mid_channels,
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            PHConv2d(n=n, in_features=mid_channels, out_features=out_channels,
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ph_n):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, ph_n)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, ph_n, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, ph_n, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, ph_n)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, ph_n):
        super(OutConv, self).__init__()
        self.conv = PHConv2d(n=ph_n, in_features=in_channels, out_features=out_channels,
                            kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, ph_n, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.ph_n = ph_n

        self.inc = DoubleConv(n_channels, 64, ph_n)
        self.down1 = Down(64, 128, ph_n)
        self.down2 = Down(128, 256, ph_n)
        self.down3 = Down(256, 512, ph_n)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, ph_n)
        self.up1 = Up(1024, 512 // factor, ph_n, bilinear)
        self.up2 = Up(512, 256 // factor, ph_n, bilinear)
        self.up3 = Up(256, 128 // factor, ph_n, bilinear)
        self.up4 = Up(128, 64, ph_n, bilinear)
        self.outc = OutConv(64, n_classes, ph_n)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)