from dataclasses import dataclass
from typing import List

from .base import Problem
from ..model import TaggingHead


@dataclass
class TaggingProblem(Problem):
    tags: List[str]

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        return TaggingHead(keep_logits=keep_logits)
