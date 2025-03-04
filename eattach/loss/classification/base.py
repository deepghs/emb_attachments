from typing import Type, Optional, Literal

import torch.nn

from ..base import register_loss as _register_base_loss


class WeightAttachment(torch.nn.Module):
    def __init__(self, num_classes, weight=None):
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes

        if weight is None:
            weight = torch.ones(num_classes, dtype=torch.float)
        self.register_buffer('weights', torch.tensor(weight))
        self.weights: Optional[torch.Tensor]

    def forward(self, loss, labels):
        return loss * self.weights[labels].to(loss.device)


class LossReduction(torch.nn.Module):
    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        torch.nn.Module.__init__(self)
        if reduction not in {'mean', 'sum'}:
            raise ValueError(f'Unknown reduction type - {reduction!r}.')
        self.reduction = reduction

    def forward(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            assert False, 'Should not reach here!'


def register_loss(name: str, module: Type[torch.nn.Module], *args, **kwargs):
    return _register_base_loss('classification', name, module, *args, **kwargs)
