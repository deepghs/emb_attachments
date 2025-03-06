from dataclasses import dataclass
from typing import Dict, Type


@dataclass
class Problem:
    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError

    @classmethod
    def load(cls, **kwargs):
        raise NotImplementedError


_PROBLEM_CLS: Dict[str, Type[Problem]] = {}


def register_problem(name: str):
    def _decorator(cls):
        _PROBLEM_CLS[name] = cls
        return cls

    return _decorator
