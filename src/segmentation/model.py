"""DeepLabv3+ variant: ResNet50 OS16, ASPP, low-level decoder, binary output.

    Standard convolutions (not separable), random weights. GroupNorm avoids the
    image-pooling BatchNorm failure for a singleton last training batch. This is
    a documented shared implementation variant, not an official pretrained model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


def block(cin, cout, kernel=3, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=(kernel//2)*dilation,
                                  dilation=dilation, bias=False),
                         nn.GroupNorm(16, cout), nn.ReLU(inplace=True))


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        base = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        base.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.low = base.layer1
        self.high = nn.Sequential(base.layer2, base.layer3, base.layer4)
        self.aspp = nn.ModuleList([block(2048, 256, 1)] +
                                  [block(2048, 256, dilation=d) for d in (6, 12, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), block(2048, 256, 1))
        self.project = block(1280, 256, 1)
        self.low_project = block(256, 48, 1)
        self.decoder = nn.Sequential(block(304, 256), block(256, 256), nn.Conv2d(256, 1, 1))

    def forward(self, x):
        low = self.low(self.stem(x))
        high = self.high(low)
        pooled = F.interpolate(self.pool(high), size=high.shape[-2:], mode='bilinear', align_corners=False)
        features = self.project(torch.cat([b(high) for b in self.aspp] + [pooled], dim=1))
        features = F.interpolate(features, size=low.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.decoder(torch.cat([features, self.low_project(low)], dim=1))
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
