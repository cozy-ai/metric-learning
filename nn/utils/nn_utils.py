import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from ..networks import FeedForward

def get_network(network, **network_args):
    """
    Build a network given the name of the network.
    """
    if network == 'resnet18':
        resnet = torchvision.models.resnet18(**network_args)
        del resnet.fc
        resnet.fc = lambda x: x
        return resnet
    elif network == 'feedforward':
        return FeedForward(**network_args)
    else:
        raise ValueError()