import torch


def dice_coefficient(y_true, y_pred):
    smooth = 1e-5
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou(y_true, y_pred):
    smooth = 1e-5
    intersection = torch.sum(y_true, y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou