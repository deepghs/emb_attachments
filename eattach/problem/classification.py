from dataclasses import dataclass
from typing import List

from .base import Problem
from ..loss import get_loss_fn
from ..model.heads import ClassificationHead


@dataclass
class ClassificationProblem(Problem):
    labels: List[str]

    def _width(self):
        return len(self.labels)

    def get_loss_fn(self, loss_fn_name: str, **kwargs):
        return get_loss_fn(
            problem='classification',
            name=loss_fn_name,
            num_classes=len(self.labels),
            **kwargs
        )

    def get_head(self, keep_logits: bool = False):
        return ClassificationHead(keep_logits=keep_logits)
