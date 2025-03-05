from dataclasses import dataclass
from typing import List

from .base import Problem
from ..model.heads import TaggingHead


@dataclass
class TaggingProblem(Problem):
    tags: List[str]

    def _width(self):
        return len(self.tags)

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        return TaggingHead(keep_logits=keep_logits)
