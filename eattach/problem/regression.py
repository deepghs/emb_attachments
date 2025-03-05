from dataclasses import dataclass
from typing import List

from .base import Problem
from ..model.heads import RegressionHead


@dataclass
class RegressionProblem(Problem):
    fields: List[str]

    def _width(self):
        return len(self.fields)

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        return RegressionHead(keep_logits=keep_logits)
