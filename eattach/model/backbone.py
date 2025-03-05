import json
import os.path
import zipfile
from dataclasses import dataclass
from typing import Optional, List, Union, Any, Dict, Type, Tuple

import torch
from hbutils.system import TemporaryDirectory
from safetensors import safe_open
from safetensors.torch import save_model, load_model
from torch import nn

from .activation import get_act_func


@dataclass
class Backbone:
    type: str
    init_params: Dict[str, Any]
    module: nn.Module

    def save(self, ckpt_file: str, extra_metadata: Dict[str, Any] = None):
        save_model(
            model=self.module,
            filename=ckpt_file,
            metadata={
                **{
                    key: json.dumps(value)
                    for key, value in (extra_metadata or {}).items()
                },
                'model_type': self.type,
                'init_params': json.dumps(self.init_params),
            }
        )

    @classmethod
    def load(cls, ckpt_file, device: Union[str, int] = 'cpu', with_metadata: bool = False) \
            -> Union['Backbone', Tuple['Backbone', dict]]:
        with safe_open(ckpt_file, 'pt') as f:
            metadata = f.metadata()

        type_ = metadata.pop('model_type')
        init_params = json.loads(metadata.pop('init_params'))
        metadata = {key: json.loads(value) for key, value in metadata.items()}
        module = _BACKBONES[type_](**init_params)
        load_model(
            model=module,
            filename=ckpt_file,
            device=device,
        )

        backbone = Backbone(
            type=type_,
            init_params=init_params,
            module=module,
        )
        if with_metadata:
            return backbone, metadata
        else:
            return backbone

    @classmethod
    def load_from_zip(cls, zip_file: str, filename: str = 'model.safetensors',
                      device: Union[str, int] = 'cpu', with_metadata: bool = False) \
            -> Union['Backbone', Tuple['Backbone', dict]]:
        with TemporaryDirectory() as td:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extract(filename, td)
                ckpt_file = os.path.join(td, filename)
                return cls.load(ckpt_file, device=device, with_metadata=with_metadata)

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
    model = Backbone.new('mlp', in_dims=1024, out_dims=10, layers=[500, 300])
    print(model)

    dummy_input = torch.randn(1, 1024)
    with torch.no_grad():
        dummy_output = model.module(dummy_input)
    print(dummy_output.shape)

    model.save('test_model.safetensors')

    model = Backbone.load('test_model.safetensors')
    print(model)
