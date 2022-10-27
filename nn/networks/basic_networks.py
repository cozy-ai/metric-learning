import torch.nn as nn
from ..utils.layer_utils import get_activation_layer

class FeedForward(nn.Module):
    """
    Simple feed forward network.
    
    Caution : activation is not applied on the last layer when n_layers>1.
    """
    def __init__(self, n_layers, in_dim, out_dim, h_dim=None, activation=None):
        super().__init__()
        if n_layers == 1:
            self.layers = nn.Linear(in_dim, out_dim)
            if activation is not None:
                activation_layer = get_activation_layer(activation)
                self.layers = nn.Sequential(*[self.layers, activation_layer])
        else:
            if h_dim is None:
                h_dim = in_dim
            if isinstance(h_dim, int):
                h_dim = [h_dim]*(n_layers-1)
            assert len(h_dim) == n_layers-1
            h_dim = [in_dim]+h_dim+[out_dim]
            layers = []
            for i in range(n_layers):
                layers.append(nn.Linear(h_dim[i], h_dim[i+1]))
                if activation is not None and i < n_layers-1:
                    layers.append(get_activation_layer(activation))
            self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)