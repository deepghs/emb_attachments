from dataclasses import dataclass
from typing import List

from .base import Problem, register_problem
from ..model import TaggingHead


@register_problem('tagging')
@dataclass
class TaggingProblem(Problem):
    tags: List[str]

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        return TaggingHead(keep_logits=keep_logits)

    def to_json(self):
        return {
            'type': 'tagging',
            'tags': self.tags,
        }

    @classmethod
    def load(cls, tags: List[str]) -> 'TaggingProblem':
        return cls(tags=tags)
