from typing import Dict, Type

from torch import nn

_ACT_FUNCS: Dict[str, Type[nn.Module]] = {
    # Basic activation functions
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,

    # Advanced/alternative activation functions
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,  # Also known as Swish
    "mish": nn.Mish,

    # Softmax variants
    "softmax": nn.Softmax,
    "log_softmax": nn.LogSoftmax,

    # Hard versions
    "hardtanh": nn.Hardtanh,
    "hardsigmoid": nn.Hardsigmoid,
    "hardswish": nn.Hardswish,
    "hardshrink": nn.Hardshrink,

    # Parameterized activation functions
    "prelu": nn.PReLU,

    # Shrinkage functions
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanhshrink": nn.Tanhshrink,

    # Specialized functions
    "glu": nn.GLU,
    "logsigmoid": nn.LogSigmoid
}


def get_act_func(name: str):
    return _ACT_FUNCS[name]()
