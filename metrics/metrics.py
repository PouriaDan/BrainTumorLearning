import torch
import numpy as np

def accuracy(gts, preds):
    if not isinstance(gts, torch.Tensor):
        gts = torch.as_tensor(np.array(gts))
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(np.array(preds))
    preds = torch.argmax(preds, dim=1)
    correct = (preds == gts).sum().item()
    total = len(gts)
    return correct / total

def precision(gts, preds):
    if not isinstance(gts, torch.Tensor):
        gts = torch.as_tensor(np.array(gts))
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(np.array(preds))
    preds = torch.argmax(preds, dim=1)
    true_positive = ((preds == 1) & (gts == 1)).sum().item()
    predicted_positive = (preds == 1).sum().item()
    if predicted_positive == 0:
        return 0.0
    return true_positive / predicted_positive

def recall(gts, preds):
    if not isinstance(gts, torch.Tensor):
        gts = torch.as_tensor(np.array(gts))
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(np.array(preds))
    preds = torch.argmax(preds, dim=1)
    true_positive = ((preds == 1) & (gts == 1)).sum().item()
    actual_positive = (gts == 1).sum().item()
    if actual_positive == 0:
        return 0.0
    return true_positive / actual_positive

def f1_score(gts, preds):
    p = precision(gts, preds)
    r = recall(gts, preds)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def confusion_matrix(gts, preds, num_classes):
    if not isinstance(gts, torch.Tensor):
        gts = torch.as_tensor(np.array(gts))
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(np.array(preds))
    preds = torch.argmax(preds, dim=1)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(gts, preds):
        matrix[t, p] += 1
    return matrix

def IoU(gts, preds):
    if not isinstance(gts, torch.Tensor):
        gts = torch.as_tensor(np.array(gts))
    if not isinstance(preds, torch.Tensor):
        preds = torch.as_tensor(np.array(preds))

    preds = torch.sigmoid(preds) > 0.5

    intersection = (gts * preds).sum().item()
    union = ((gts + preds) > 0).sum().item()

    if union == 0:
        return 0.0

    return intersection / union



