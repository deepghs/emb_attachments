from torch import nn


class TrainModel(nn.Module):
    def __init__(self, encoder: nn.Module, backbone: nn.Module):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.backbone = backbone

    def forward(self, x):
        embedding = self.encoder(x)
        return self.backbone(embedding)


class FullModel(nn.Module):
    def __init__(self, encoder: nn.Module, backbone: nn.Module, head: nn.Module):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.backbone(embedding)
        return embedding, logits, self.head(logits)


class BackboneWithHead(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        logits = self.backbone(x)
        return logits, self.head(logits)
