from typing import Mapping, Tuple, Type, Dict, List, Literal

import torch.nn

_KNOWN_LOSSES: Dict[str, Dict[str, Tuple[Type[torch.nn.Module], Tuple, Mapping]]] = {}


def register_loss(problem: str, name: str, module: Type[torch.nn.Module], *args, **kwargs):
    if problem not in _KNOWN_LOSSES:
        _KNOWN_LOSSES[problem] = {}
    if name not in _KNOWN_LOSSES[problem]:
        _KNOWN_LOSSES[problem][name] = (module, args, kwargs)
    else:
        raise ValueError(f'Loss {name!r} for {problem} problem already exist!')


def get_loss_fn(problem: str, name: str, num_classes: int, weight, reduction: Literal['mean', 'sum'] = 'mean',
                **loss_args):
    if problem in _KNOWN_LOSSES:
        if name in _KNOWN_LOSSES[problem]:
            module, args, kwargs = _KNOWN_LOSSES[problem][name]

            # noinspection PyArgumentList
            return module(num_classes, *args, reduction=reduction, weight=weight, **kwargs, **loss_args)
        else:
            raise ValueError(f'Loss function should be one of the {list_losses()!r}, but {name!r} found.')
    else:
        raise ValueError(f'Unknown problem type for loss function - {problem!r}.')


def list_losses(problem: str) -> List[str]:
    if problem in _KNOWN_LOSSES:
        return sorted(_KNOWN_LOSSES[problem].keys())
    else:
        return []
