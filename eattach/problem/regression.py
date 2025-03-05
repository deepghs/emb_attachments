from dataclasses import dataclass
from typing import List

from .base import Problem


@dataclass
class RegressionProblem(Problem):
    fields: List[str]

    def _width(self):
        return len(self.fields)

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError
