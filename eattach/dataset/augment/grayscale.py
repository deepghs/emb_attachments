import random
from functools import wraps


def prob_op(prob, op):
    @wraps(op)
    def _func(img):
        if random.random() < prob:
            return op(img)
        else:
            return img

    return _func


def _to_greyscale(img):
    origin_mode = img.mode
    return img.convert('L').convert(origin_mode)


def prob_greyscale(prob: float = 0.5):
    return prob_op(prob, _to_greyscale)
