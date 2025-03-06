from dataclasses import dataclass
from typing import List, Tuple

from .base import Problem
from ..model import RegressionHead


@dataclass
class RegressionProblem(Problem):
    fields: List[Tuple[str, float, float]]

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        means, stds = [], []
        for _, mean, std in self.fields:
            means.append(mean)
            stds.append(std)
        return RegressionHead(means=means, stds=stds, keep_logits=keep_logits)

    def to_json(self):
        return {
            'type': 'regression',
            'fields': [
                {
                    'name': field_name,
                    'mean': mean,
                    'std': std,
                }
                for field_name, mean, std in self.fields
            ]
        }
