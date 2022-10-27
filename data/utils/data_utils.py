import torchvision
from ..wrappers import PairDatasetWrapper, TwinDatasetWrapper

def get_dataset(dataset_name, **kwargs):
    """
    Return dataset given the name of the dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar-10':
        return torchvision.datasets.CIFAR10(**kwargs)
    elif dataset_name == 'cifar-100':
        return torchvision.datasets.CIFAR100(**kwargs)
    else:
        raise ValueError()
        
def get_dataset_wrapper(dataset_wrapper):
    """
    Return dataset wrapper class given the name of the wrapper.
    """
    dataset_wrapper = dataset_wrapper.lower()
    if dataset_wrapper == 'pair':
        return PairDatasetWrapper
    elif dataset_wrapper == 'twin':
        return TwinDatasetWrapper
    else:
        raise ValueError()