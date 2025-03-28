from dataclasses import dataclass
from typing import List

from .base import Problem, register_problem
from ..loss import get_loss_fn
from ..model import ClassificationHead


@register_problem('classification')
@dataclass
class ClassificationProblem(Problem):
    labels: List[str]

    def get_loss_fn(self, loss_fn_name: str, **kwargs):
        return get_loss_fn(
            problem='classification',
            name=loss_fn_name,
            num_classes=len(self.labels),
            **kwargs
        )

    def get_head(self, keep_logits: bool = False):
        return ClassificationHead(keep_logits=keep_logits)

    def to_json(self):
        return {
            'type': 'classification',
            'labels': self.labels,
        }

    @classmethod
    def load(cls, labels: List[str]) -> 'ClassificationProblem':
        return cls(labels=labels)
