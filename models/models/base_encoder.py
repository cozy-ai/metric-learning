import torch.nn as nn
from ...nn import get_network

class BaseEncoder(nn.Module):
    """
    Basic encoder with backbone and neck networks.
    """
    def __init__(self, backbone, backbone_args, neck, neck_args):
        super().__init__()
        self.backbone = get_network(backbone, **backbone_args)
        self.neck = get_network(neck, **neck_args)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x
