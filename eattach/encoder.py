import dataclasses
from functools import partial
from typing import Callable, Dict, Any

import numpy as np
from imgutils.tagging import get_wd14_tags

FnEncodeTyping = Callable[[Any], np.ndarray]


@dataclasses.dataclass
class Encoder:
    encode_system: str
    model_name: str
    fn_encode: FnEncodeTyping

    def __call__(self, obj) -> np.ndarray:
        return self.fn_encode(obj)

    def to_json(self):
        return {
            'encode_system': self.encode_system,
            'model_name': self.model_name,
        }


_ENCODE_SYSTEMS: Dict[str, Callable[[str], FnEncodeTyping]] = {}


def register_encode_system(name: str):
    def _decorator(func):
        _ENCODE_SYSTEMS[name] = func
        return func

    return _decorator


@register_encode_system('imgutils.tagging.wd14')
def _get_wd14_encoder_fn(model_name: str):
    return partial(get_wd14_tags, model_name=model_name, fmt='embedding')


def get_encoder(encode_system: str, model_name: str) -> Encoder:
    fn_encode = _ENCODE_SYSTEMS[encode_system](model_name)
    return Encoder(
        encode_system=encode_system,
        model_name=model_name,
        fn_encode=fn_encode,
    )
