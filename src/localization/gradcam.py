"""Positive-logit Grad-CAM for the four configured classifiers."""
import torch
from torch.nn import functional as F


class GradCAM:
    """Use a forward hook and autograd.grad without updating classifier weights.

    DeiT uses tokens before the final attention mixing, not final patch tokens.
    MiT's encoder output is already spatial. No ground-truth input is accepted.
    """
    def __init__(self, model, name):
        self.model, self.name = model, name
        if name in ('resnet18', 'resnet50'):
            self.layer_name = 'backbone.layer4'
            layer = model.backbone.layer4
        elif name == 'deit_small':
            self.layer_name = 'backbone.blocks[-1].norm1'
            layer = model.backbone.blocks[-1].norm1
        elif name == 'mit_b1':
            self.layer_name = 'encoder.last_hidden_state'
            layer = model.encoder
        else:
            raise ValueError('No configured Grad-CAM adapter for this classifier')
        self.activation = None
        self.handle = layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.activation = output.last_hidden_state if self.name == 'mit_b1' else output

    def _spatial(self, tensor, height, width):
        if self.name == 'deit_small':
            ph, pw = self.model.backbone.patch_embed.patch_size
            gh, gw = height // ph, width // pw
            prefix = self.model.backbone.num_prefix_tokens
            tensor = tensor[:, prefix:, :]
            if tensor.shape[1] != gh * gw:
                raise ValueError('DeiT token grid does not match input geometry')
            return tensor.reshape(tensor.shape[0], gh, gw, tensor.shape[-1]).permute(0, 3, 1, 2)
        if tensor.ndim != 4:
            raise ValueError('Expected NCHW spatial activation')
        return tensor

    def __call__(self, images):
        self.model.eval()
        self.activation = None
        with torch.enable_grad():
            images = images.detach().requires_grad_(True)
            logits = self.model(images)
            if self.activation is None or logits.shape != (images.shape[0],):
                raise ValueError('Expected captured features and one logit per image')
            gradient, = torch.autograd.grad(logits.sum(), self.activation)
            activation = self._spatial(self.activation, *images.shape[-2:])
            gradient = self._spatial(gradient, *images.shape[-2:])
            if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite features or gradients')
            magnitude = gradient.abs().flatten(1).sum(1)
            if (magnitude == 0).any():
                raise ValueError('Zero spatial gradients: review target layer/checkpoint')
            signed = (gradient.mean((-2, -1), keepdim=True) * activation).sum(1)
            raw = F.interpolate(signed.relu()[:, None], size=images.shape[-2:],
                                mode='bilinear', align_corners=False)[:, 0]
            if not torch.isfinite(raw).all() or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite attribution output')
            result = {'logits': logits.detach(), 'raw': raw.detach(),
                      'signed_native': signed.detach(), 'gradient_l1': magnitude.detach()}
        self.activation = None
        return result

    def close(self):
        self.handle.remove()
        self.activation = None

