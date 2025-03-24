from torch import nn

_FN_NORMS = {}


def register_fn_norm(norm_type: str):
    def _decorator(fn):
        _FN_NORMS[norm_type] = fn
        return fn

    return _decorator


@register_fn_norm('batch')
def create_batch_norm(width: int):
    return nn.BatchNorm1d(width)


@register_fn_norm('layer')
def create_layer_norm(width: int):
    return nn.LayerNorm(width)


class GroupNormWrapper(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNormWrapper, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.norm(x)
        x = x.squeeze(-1)
        return x


@register_fn_norm('group')
def create_group_norm(width: int, groups: int = 32):
    groups = min(groups, width)
    return GroupNormWrapper(groups, width)


def create_norm(norm_type: str, width: int, **kwargs):
    if norm_type == 'none':
        return None
    else:
        return _FN_NORMS[norm_type](width, **kwargs)
