import torch
import numpy as np

def Dice(preds, targets, smooth=1e-6, threshold=None):
    if preds.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    # Apply thresholding for binary or multi-class case
    if threshold is not None:
        preds = (preds > threshold).float()
    # Flatten tensors except for batch and channel dimensions
    preds = preds.flatten(2)  # (B, C, H*W)
    targets = targets.flatten(2)
    intersection = (preds * targets).sum(dim=-1)
    union = preds.sum(dim=-1) + targets.sum(dim=-1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()



def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()
    return (2.0 * intersection + smooth) / (total + smooth)

def precision(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    fp = preds.sum() - tp
    return (tp + smooth) / (tp + fp + smooth)

def recall(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    fn = targets.sum() - tp
    return (tp + smooth) / (tp + fn + smooth)

def specificity(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float()
    tn = ((1 - preds) * (1 - targets)).sum()
    fp = preds.sum() - (preds * targets).sum()
    return (tn + smooth) / (tn + fp + smooth)

def f1_score(precision, recall, beta=1.0, smooth=1e-6):
    return (1 + beta**2) * (precision * recall + smooth) / (beta**2 * precision + recall + smooth)

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2))

def binary_iou(pred, target):
    pred, target = pred.view(-1), target.view(-1)
    intersection = (pred & target).float().sum().item()
    union = pred.float().sum().item() + target.float().sum().item() - intersection
    return intersection / union if union != 0 else np.nan

def binary_dice(pred, target):
    pred, target = pred.view(-1), target.view(-1)
    intersection = (pred & target).float().sum().item()
    return (2. * intersection) / (pred.float().sum().item() + target.float().sum().item() + 1e-8)

def binary_f1(pred, target):
    pred, target = pred.view(-1), target.view(-1)
    tp = (pred & target).sum().item()
    fp = (pred & (~target)).sum().item()
    fn = ((~pred) & target).sum().item()
    return (2 * tp) / (2 * tp + fp + fn + 1e-8) if (tp + fp + fn) != 0 else np.nan
