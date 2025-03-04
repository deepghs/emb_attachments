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


class MultiLabelClassificationHead(nn.Module):
    def __init__(self, keep_logits: bool = False):
        nn.Module.__init__(self)
        self.sigmoid = nn.Sigmoid()
        self.split_head = SplitHead(keep_logits)

    def forward(self, x):
        pred = self.sigmoid(x)
        return self.split_head(pred, x)


class RegressionHead(nn.Module):
    def __init__(self, keep_logits: bool = False):
        nn.Module.__init__(self)
        self.split_head = SplitHead(keep_logits)

    def forward(self, x):
        pred = x
        return self.split_head(pred, x)


_PROBLEMS = {
    'classification': ClassificationHead,
    'multi_label': MultiLabelClassificationHead,
    'regression': RegressionHead,
}


def get_problem_head(problem: str, keep_logits: bool = False) -> nn.Module:
    return _PROBLEMS[problem](keep_logits=keep_logits)


if __name__ == '__main__':
    print(get_problem_head('classification'))
