import json
from dataclasses import dataclass
from typing import Optional, List, Union, Any, Dict, Type

from safetensors import safe_open
from safetensors.torch import save_model, load_model
from torch import nn

from .activation import get_act_func


@dataclass
class Backbone:
    type: str
    init_params: Dict[str, Any]
    module: nn.Module

    def save(self, ckpt_file: str):
        save_model(
            model=self.module,
            filename=ckpt_file,
            metadata={
                'model_type': self.type,
                'init_params': json.dumps(self.init_params),
            }
        )

    @classmethod
    def load(cls, ckpt_file, device: Union[str, int] = 'cpu') -> 'Backbone':
        with safe_open(ckpt_file, 'pt') as f:
            metadata = f.metadata()

        type_ = metadata['model_type']
        init_params = json.loads(metadata['init_params'])
        module = _BACKBONES[type_](**init_params)
        load_model(
            model=module,
            filename=ckpt_file,
            device=device,
        )

        return Backbone(
            type=type_,
            init_params=init_params,
            module=module,
        )

    @classmethod
    def new(cls, type_: str, **init_params) -> 'Backbone':
        module = _BACKBONES[type_](**init_params)
        return Backbone(
            type=type_,
            init_params=init_params,
            module=module,
        )


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


if __name__ == '__main__':
    model = Backbone.new('mlp', layers=[500, 300])
    print(model)
    model.save('test_model.safetensors')

    model = Backbone.load('test_model.safetensors')
    print(model)
