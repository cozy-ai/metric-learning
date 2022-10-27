from .criterions import PairwiseCE, InfoNCE
from .metrics import l1, dot, cosine, ap, batch_ap

def get_criterion(criterion):
    """
    Return a criterion (callable), given the name of the criterion.
    """
    criterion = criterion.lower()
    if criterion == 'pairwise_ce':
        return PairwiseCE()
    elif criterion == 'info_nce':
        return InfoNCE()
    else:
        raise ValueError()

def get_dist_metric(dist_metric, dist_linear=None):
    """
    Return a distance metric function, given the name of the distance metric function.
    
    Compute distance metric on LAST dimension.
    """
    dist_metric = dist_metric.lower()
    aggregate = dist_linear is None
    
    if dist_metric == 'l1-dist':
        metric = l1
    elif dist_metric == 'dot-sim':
        metric = dot
    elif dist_metric == 'cosine-sim':
        metric = cosine
    else:
        raise ValueError()
    fn = lambda *args, **kwargs: dist_linear(metric(*args, **kwargs)) if dist_linear is not None else metric(*args, **kwargs)
    
    
    return lambda out1, out2: fn(out1,out2, aggregate=aggregate)

def get_eval_metric(eval_metric, model_wrapper=None):
    """
    Return a eval metric function, given the name of the eval metric function.
    """
    eval_metric = eval_metric.lower()
    if eval_metric == 'batch_ap':
        metric = batch_ap
    elif eval_metric == 'ap':
        metric = ap
    elif eval_metric[:10] == 'ap@':
        k = int(eval_metric[10:])
        metric = lambda *args, **kwargs: ap(k=k, *args, **kwargs)
    elif eval_metric[:16] == 'batch_ap@':
        k = int(eval_metric[16:])
        metric = lambda *args, **kwargs: batch_ap(k=k, *args, **kwargs)
    else:
        raise ValueError()
        
    return metric