import torch.nn as nn

def get_activation_layer(activation):
    """
    Return an activation layer instance given the name of the activation.
    """
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
