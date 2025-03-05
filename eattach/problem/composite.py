from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import torch
from hbutils.random import global_seed
from torch import nn

from .base import Problem
from ..model.heads import SplitHead


class CompositeLossFn(nn.Module):
    def __init__(self, items: List[Tuple[str, int, nn.Module, float]], **kwargs):
        _ = kwargs
        nn.Module.__init__(self)

        _last_offset = 0
        self._slices = {}
        self._keys = []
        _submodules = {}
        _weights = []
        for key, width, submodule, weight in items:
            self._keys.append(key)
            self._slices[key] = slice(_last_offset, _last_offset + width)
            _submodules[key] = submodule
            _weights.append(weight)
            _last_offset += width

        self.submodules = nn.ModuleDict(_submodules)
        _weights = torch.tensor(_weights)
        self.register_buffer('weights', _weights)
        self.weights: torch.Tensor

    def forward(self, x, expected):
        values = []
        for key in self._keys:
            values.append(self.submodules[key](x[..., self._slices[key]], expected[key]))

        values = torch.stack(values, dim=-1)
        return (self.weights * values).sum(dim=-1)


class CompositeHead(nn.Module):
    def __init__(self, items: List[Tuple[str, int, nn.Module]], keep_logits: bool = False):
        nn.Module.__init__(self)

        _last_offset = 0
        self._slices = {}
        self._keys = []
        _submodules = {}
        for key, width, submodule in items:
            self._keys.append(key)
            self._slices[key] = slice(_last_offset, _last_offset + width)
            _submodules[key] = submodule
            _last_offset += width

        self.submodules = nn.ModuleDict(_submodules)
        self.split_head = SplitHead(keep_logits=keep_logits)

    def forward(self, x):
        pred = {}
        for key in self._keys:
            pred[key] = self.submodules[key](x[..., self._slices[key]])
        return self.split_head(pred, x)


@dataclass
class CompositeProblem(Problem):
    problems: Dict[str, Problem]

    def _width(self):
        return sum([problem.width for _, problem in self.problems])

    def get_loss_fn(self, loss_weights: Optional[Dict[str, float]] = None,
                    problems_cfg: Optional[Dict[str, dict]] = None, reduction: str = 'mean', **kwargs):
        problems_cfg = dict(problems_cfg or {})
        loss_weights = dict(loss_weights or {})
        items = []
        for key, problem in self.problems.items():
            items.append((
                key,
                problem.width,
                problem.get_loss_fn(**{'reduction': reduction, **kwargs, **dict(problems_cfg.get(key) or {})}),
                loss_weights.get(key, 1.0)
            ))
        return CompositeLossFn(items, **kwargs)

    def get_head(self, keep_logits: bool = False):
        items = []
        for key, problem in self.problems.items():
            items.append((key, problem.width, problem.get_head(keep_logits=False)))

        return CompositeHead(items, keep_logits=keep_logits)


if __name__ == '__main__':
    from .classification import ClassificationProblem

    global_seed(0)

    p = CompositeProblem(
        problems={
            'p1': ClassificationProblem(['a', 'b', 'c']),
            'p2': ClassificationProblem(['a', 'b', 'c', 'd']),
        }
    )
    print(p)
    loss_fn = p.get_loss_fn(
        problems_cfg={
            'p1': {'loss_fn_name': 'focal'},
            'p2': {'loss_fn_name': 'focal'},
        },
        loss_weights={'p1': 0.5},
        reduction='none',
    )
    print(loss_fn)

    dummy_input = torch.randn(10, 7)
    v = loss_fn(
        dummy_input,
        {
            'p1': torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            'p2': torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
        },
    )
    print(v)
    print(v.mean())

    head = p.get_head()
    print(head(dummy_input))
