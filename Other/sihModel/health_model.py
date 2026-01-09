# health_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetHealthMap(nn.Module):
    """U-Net model for predicting RGB health maps from satellite imagery."""
    def __init__(self, in_ch=21, base=32):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*8, base*16)

        # decoder
        self.up4 = UpBlock(base*16, base*8)
        self.up3 = UpBlock(base*8, base*4)
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)

        # RGB health map head (3 channels for RGB)
        self.head_health = nn.Sequential(
            nn.Conv2d(base, base//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 3, kernel_size=1),
            nn.Sigmoid()  # Output in 0-1 range for RGB
        )

        # Optional: regression head for yield (if still needed)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base*16, base*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base*8, 1)
        )

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)       # base
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        health_map = self.head_health(d1)   # (B, 3, H, W) RGB health map
        y = self.regressor(b)               # (B, 1) yield prediction

        return health_map, y

class UNetHealthMapOnly(nn.Module):
    """U-Net model for predicting only RGB health maps (no yield regression)."""
    def __init__(self, in_ch=21, base=32):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*8, base*16)

        # decoder
        self.up4 = UpBlock(base*16, base*8)
        self.up3 = UpBlock(base*8, base*4)
        self.up2 = UpBlock(base*4, base*2)
        self.up1 = UpBlock(base*2, base)

        # RGB health map head
        self.head_health = nn.Sequential(
            nn.Conv2d(base, base//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base//2, 3, kernel_size=1),
            nn.Sigmoid()  # Output in 0-1 range for RGB
        )

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)       # base
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        health_map = self.head_health(d1)   # (B, 3, H, W) RGB health map

        return health_map