import copy

from .base import _PROBLEM_CLS, Problem


def load_problem(config) -> Problem:
    config: dict = copy.deepcopy(config)
    type_ = config.pop('type')
    return _PROBLEM_CLS[type_].load(**config)
