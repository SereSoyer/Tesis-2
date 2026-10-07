"""Binary classification epochs. No spatial targets are consumed."""
import time
import torch
from src.evaluation import ClassificationMetrics


def run_epoch(model, loader, device, threshold=0.5, optimizer=None, expected_shape=None):
    training = optimizer is not None
    model.train(training)
    metrics = ClassificationMetrics(threshold)
    total_loss, count = 0.0, 0
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            if expected_shape is not None and tuple(images.shape[1:]) != tuple(expected_shape):
                raise ValueError('Input shape differs from configuration; no implicit resize')
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape != labels.shape:
                raise ValueError('Expected one logit per image')
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if training:
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise ValueError('Missing or nonfinite gradients')
                optimizer.step()
            metrics.update(labels, logits.detach().sigmoid())
            count += labels.numel()
            total_loss += loss.item() * labels.numel()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    report = metrics.compute()
    report.update(loss=total_loss/count, seconds=time.perf_counter()-start)
    return report
