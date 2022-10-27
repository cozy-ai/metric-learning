import torch.optim as optim
from collections.abc import Iterable

def update_params(model, optimizer_args):
    if isinstance(optimizer_args['params'], str):
        optimizer_args.update({'params': getattr(model, optimizer_args['params']).parameters()})
    if isinstance(optimizer_args['params'], Iterable):
        for param_group in optimizer_args['params']:
            if isinstance(param_group['params'], str):
                param_group.update({'params': getattr(model, param_group['params']).parameters()})

def get_optimizer(optimizer, model, **optimizer_args):
    """
    Return an optimizer given the name of the optimizer.
    """
    optimizer = optimizer.lower()
    update_params(model, optimizer_args)
    if optimizer == 'adam':
        return optim.Adam(**optimizer_args)
    else:
        raise ValueError()
        
def get_scheduler(scheduler, optimizer, warm_up=0, **scheduler_args):
    """
    Return scheduler
    """
    scheduler=scheduler.lower()
    if scheduler == 'cosineannealinglr':
        out = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        out.warm_up = warm_up
    else:
        raise ValueError()