from dataclasses import dataclass
from pprint import pprint
from typing import Optional, List, Union, Any, Dict, Type

from torch import nn

from .activation import get_act_func


@dataclass
class Backbone:
    type: str
    init_params: Dict[str, Any]
    module: nn.Module

    def to_json(self):
        return {
            **self.init_params,
            'type': self.type,
        }


_BACKBONES: Dict[str, Type[nn.Module]] = {}


def register_backbone(name: str):
    def _decorator(module_cls: Type[nn.Module]):
        _BACKBONES[name] = module_cls
        return module_cls

    return _decorator


@register_backbone('mlp')
class MLP(nn.Module):
    def __init__(self, in_dims: int = 1024, out_dims: int = 1,
                 layers: Optional[List[Union[int, float]]] = None, act_func: str = 'relu',
                 dropout_rate: float = 0.2, **kwargs):
        nn.Module.__init__(self)
        _ = kwargs
        layers = [
            x if isinstance(x, int) else int(round(x * in_dims))
            for x in (layers or [])
        ]
        _layers = []

        layers = [in_dims, *layers]
        for prev_layer, next_layer in zip(layers[:-1], layers[1:]):
            _layers.append(nn.Linear(prev_layer, next_layer))
            _layers.append(get_act_func(act_func))
            if dropout_rate > 0:
                _layers.append(nn.Dropout(p=dropout_rate))
        _layers.append(nn.Linear(layers[-1], out_dims))
        self._layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self._layers(x)


def create_backbone(type: str, **init_params) -> Backbone:
    return Backbone(
        type=type,
        init_params=init_params,
        module=_BACKBONES[type](**init_params),
    )


if __name__ == '__main__':
    model = create_backbone('mlp', layers=[500, 300])
    print(model)
    pprint(model.to_json())
