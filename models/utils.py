from .models import BaseEncoder
from .wrappers import PairContrastiveWrapper

def get_model_wrapper(model_wrapper):
    """
    Return a model wrapper class given the name of the model.
    """
    model_wrapper = model_wrapper.lower()
    if model_wrapper == 'pair_contrastive':
        return PairContrastiveWrapper
    else:
        raise ValueError()
        
def get_model(model, *args, **kwargs):
    """
    Return a model instance given the name of the model.
    """
    model = model.lower()
    if model == 'base_encoder':
        return BaseEncoder(*args, **kwargs)
    else:
        raise ValueError()