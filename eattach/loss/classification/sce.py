from typing import Literal

import torch
import torch.nn.functional as F

from .base import register_loss


class SCELoss(torch.nn.Module):
    def __init__(self, num_classes, alpha=0.1, beta=1.0, reduction: Literal['mean', 'sum'] = 'mean', weight=None,
                 **kwargs):
        _ = kwargs
        super(SCELoss, self).__init__()
        _ = reduction, weight
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


register_loss('sce', SCELoss)
