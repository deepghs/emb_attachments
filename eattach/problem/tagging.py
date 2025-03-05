from dataclasses import dataclass
from typing import List

from .base import Problem


@dataclass
class TaggingProblem(Problem):
    tags: List[str]

    def _width(self):
        return len(self.tags)

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError
