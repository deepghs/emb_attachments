from typing import Mapping, Union

from PIL import Image

from eattach.model import Backbone


class BaseLogger:
    def __init__(self, workdir, **kwargs):
        _ = kwargs
        self.workdir = workdir

    def tb_train_log(self, global_step, metrics: Mapping[str, Union[float, Image.Image]]):
        raise NotImplementedError

    def tb_eval_log(self, global_step, model: Backbone, metrics: Mapping[str, Union[float, Image.Image]]):
        raise NotImplementedError
