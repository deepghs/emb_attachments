from dataclasses import dataclass


@dataclass
class Problem:
    def _width(self):
        raise NotImplementedError

    @property
    def width(self) -> int:
        return self._width()

    def get_loss_fn(self, **kwargs):
        raise NotImplementedError

    def get_head(self, keep_logits: bool = False):
        raise NotImplementedError
