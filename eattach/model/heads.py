import torch
from torch import nn


class SplitHead(nn.Module):
    def __init__(self, keep_logits: bool = False):
        nn.Module.__init__(self)
        self.keep_logits = keep_logits

    def forward(self, pred, x):
        if self.keep_logits:
            return pred, x
        else:
            return pred


class ClassificationHead(nn.Module):
    def __init__(self, keep_logits: bool = False):
        nn.Module.__init__(self)
        self.softmax = nn.Softmax()
        self.split_head = SplitHead(keep_logits)

    def forward(self, x):
        pred = self.softmax(x)
        return self.split_head(pred, x)


class TaggingHead(nn.Module):
    def __init__(self, keep_logits: bool = False):
        nn.Module.__init__(self)
        self.sigmoid = nn.Sigmoid()
        self.split_head = SplitHead(keep_logits)

    def forward(self, x):
        pred = self.sigmoid(x)
        return self.split_head(pred, x)


class RegressionHead(nn.Module):
    def __init__(self, means, stds, keep_logits: bool = False):
        nn.Module.__init__(self)
        self.split_head = SplitHead(keep_logits)

        means = torch.tensor(means)
        self.register_buffer('means', means)
        self.means: torch.Tensor

        stds = torch.tensor(stds)
        self.register_buffer('stds', stds)
        self.stds: torch.Tensor

    def forward(self, x):
        # N(0, 1) --> N(mean, std)
        pred = x * self.stds + self.means
        return self.split_head(pred, x)
