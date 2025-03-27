import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = out_channels
        init_channels = out_channels // ratio
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att

class GAUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAUNet, self).__init__()

        self.encoder1 = GhostModule(in_channels, 64)
        self.encoder2 = GhostModule(64, 128)
        self.encoder3 = GhostModule(128, 256)
        self.encoder4 = GhostModule(256, 512)

        self.decoder1 = GhostModule(512 + 256, 256)
        self.decoder2 = GhostModule(256 + 128, 128)
        self.decoder3 = GhostModule(128 + 64, 64)

        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        d1 = self.decoder1(torch.cat([self.up(e4), e3], dim=1))
        d1 = self.cbam1(d1)
        d2 = self.decoder2(torch.cat([self.up(d1), e2], dim=1))
        d2 = self.cbam2(d2)
        d3 = self.decoder3(torch.cat([self.up(d2), e1], dim=1))
        d3 = self.cbam3(d3)

        output = self.final_conv(d3)

        return output
