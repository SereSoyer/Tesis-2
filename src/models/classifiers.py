"""Four classifiers with a common one-logit-per-image interface."""
import torch
from torch import nn

MODELS = ('resnet18', 'resnet50', 'deit_small', 'mit_b1')

class MiTClassifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel
        config = SegformerConfig(num_channels=channels, depths=[2,2,2,2],
                                 hidden_sizes=[64,128,320,512], num_attention_heads=[1,2,5,8],
                                 sr_ratios=[8,4,2,1], reshape_last_stage=True)
        self.encoder = SegformerModel(config)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        features = self.encoder(pixel_values=x).last_hidden_state
        return self.classifier(features.mean(dim=(-2,-1))).squeeze(-1)

class BinaryClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

def build_model(name, in_channels, image_size=128, pretrained=False):
    if name not in MODELS or in_channels < 1:
        raise ValueError('Invalid model or channel count')
    if pretrained:
        raise ValueError('Pretraining and RGB weight adaptation are not yet approved; use pretrained=False for smoke checks')
    if name == 'mit_b1':
        return MiTClassifier(in_channels)
    import timm
    kwargs = dict(pretrained=False, in_chans=in_channels, num_classes=1)
    if name == 'deit_small':
        if image_size % 16:
            raise ValueError('DeiT requires image_size divisible by 16')
        kwargs['img_size'] = image_size
        name = 'deit_small_patch16_224'  # Non-distilled DeiT; configured resolution is explicit.
    return BinaryClassifier(timm.create_model(name, **kwargs))
